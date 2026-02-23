

import os
from typing import Dict, Any, List, Optional

import pandas as pd
from huggingface_hub import hf_hub_download, HfFolder
from huggingface_hub.errors import (
    GatedRepoError,
    RepositoryNotFoundError,
    EntryNotFoundError,
)

from .base_benchmark import BaseBenchmark
from .GAIA.scoring_utils import question_scorer



GAIA_REPO_ID = "gaia-benchmark/GAIA"
GAIA_DEFAULT_CONFIG = "2023_all"

GAIA_LEVELS = {
    "2023": [1, 2, 3],
}



class GaiaBenchmark(BaseBenchmark):
    """GAIA benchmark implementation (Parquet-backed, HF-native)."""

    def __init__(
        self,
        agent_dir: str,
        config: Dict[str, Any],
        benchmark_name: str = "gaia",
    ):
        self.benchmark_name = benchmark_name
        self.setup_script = None
        self.requires_sandbox = False

        super().__init__(
            agent_dir,
            config,
            requires_sandbox=self.requires_sandbox,
            setup_script=self.setup_script,
        )

        dataset = self._load_gaia_dataset(
            config_split=GAIA_DEFAULT_CONFIG,
            split="validation",
        )

        self.benchmark: Dict[str, Dict[str, Any]] = {}

        for record in dataset:
            task_id = record["task_id"]
            self.benchmark[task_id] = record

            if record.get("file_name"):
                self.benchmark[task_id]["files"] = {
                    record["file_name"]: record.get("file_path", "")
                }


    def evaluate_output(
        self,
        agent_output: Dict[str, Any],
        run_id: str,
    ) -> Dict[str, Any]:
        """Evaluate agent outputs using GAIA scorer."""

        eval_results: Dict[str, Any] = {}

        for task_id, agent_answer in agent_output.items():
            answer_payload = agent_answer

            if isinstance(agent_answer, dict):
                answer_payload = agent_answer.get(
                    "answer",
                    agent_answer.get("raw_response"),
                )

            gt_answer = self.benchmark[task_id]["Final answer"]
            score, explanation = question_scorer(
                str(answer_payload),
                str(gt_answer),
            )

            eval_results[task_id] = {
                "score": score,
                "explanation": explanation,
            }

        return eval_results


    def get_metrics(
        self,
        eval_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute overall and per-level GAIA accuracy metrics"""


        overall_correct = 0

        level_correct = {"level_1": 0, "level_2": 0, "level_3": 0}
        level_total = {"level_1": 0, "level_2": 0, "level_3": 0}

        for task_id, record in self.benchmark.items():
            level_key = f"level_{record['Level']}"

            if level_key in level_total:
                level_total[level_key] += 1

            if task_id not in eval_results:
                continue

            if int(eval_results[task_id]["score"]) > 0:
                overall_correct += 1
                if level_key in level_correct:
                    level_correct[level_key] += 1

        successful_tasks = [
            tid for tid, r in eval_results.items()
            if int(r["score"]) > 0
        ]

        failed_tasks = [
            tid for tid, r in eval_results.items()
            if int(r["score"]) == 0
        ]

        return {
            "accuracy": overall_correct / max(len(eval_results), 1),
            "level_1_accuracy": (
                level_correct["level_1"] / level_total["level_1"]
                if level_total["level_1"] > 0 else None
            ),
            "level_2_accuracy": (
                level_correct["level_2"] / level_total["level_2"]
                if level_total["level_2"] > 0 else None
            ),
            "level_3_accuracy": (
                level_correct["level_3"] / level_total["level_3"]
                if level_total["level_3"] > 0 else None
            ),
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
        }

    # ============================================================
    # Dataset Loading (Parquet)
    # ============================================================

    def _load_gaia_dataset(
        self,
        config_split: str,
        split: str,
    ) -> List[Dict[str, Any]]:
        """Load GAIA dataset using Parquet metadata."""

        try:
            year, level_suffix = config_split.split("_", 1)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid GAIA config '{config_split}'."
            ) from exc

        if level_suffix == "all":
            levels = GAIA_LEVELS.get(year, [])
        else:
            try:
                levels = [int(level_suffix.replace("level", ""))]
            except ValueError as exc:
                raise RuntimeError(
                    f"Unsupported GAIA split '{config_split}'."
                ) from exc

        if not levels:
            raise RuntimeError(
                f"No GAIA levels available for year '{year}'."
            )

        token = self._resolve_hf_token()
        if token is None:
            raise RuntimeError(
                "GAIA access requires a Hugging Face token. "
                "Run `huggingface-cli login`."
            )

        metadata_path = self._download_gaia_file(
            f"{year}/{split}/metadata.parquet",
            token,
        )

        df = pd.read_parquet(metadata_path)
        records = df.to_dict(orient="records")

        filtered_records: List[Dict[str, Any]] = {}
        attachment_cache: Dict[str, str] = {"": ""}

        output_records: List[Dict[str, Any]] = []

        for record in records:
            try:
                level = int(record.get("Level"))
            except (TypeError, ValueError):
                continue

            if level not in levels:
                continue

            file_path = record.get("file_path") or ""
            file_name = record.get("file_name") or ""

            if file_path:
                if file_path not in attachment_cache:
                    attachment_cache[file_path] = self._download_gaia_file(
                        file_path,
                        token,
                    )
                record["file_path"] = attachment_cache[file_path]
            else:
                record["file_path"] = ""

            output_records.append(record)

        if not output_records:
            raise RuntimeError(
                f"No GAIA records loaded for split '{split}'."
            )

        return output_records

    # ============================================================
    # HF Utilities
    # ============================================================

    def _download_gaia_file(
        self,
        relative_path: str,
        token: str,
    ) -> str:
        """Download a GAIA dataset file from Hugging Face"""

        try:
            return hf_hub_download(
                repo_id=GAIA_REPO_ID,
                filename=relative_path,
                repo_type="dataset",
                token=token,
            )
        except GatedRepoError as exc:
            raise RuntimeError(
                "GAIA dataset is gated. "
                "Accept the license on Hugging Face."
            ) from exc
        except EntryNotFoundError as exc:
            raise RuntimeError(
                f"GAIA file '{relative_path}' not found."
            ) from exc
        except RepositoryNotFoundError as exc:
            raise RuntimeError(
                f"Dataset {GAIA_REPO_ID} not available."
            ) from exc

    def _resolve_hf_token(self) -> Optional[str]:
        """Resolve Hugging Face token from environment or local cache"""

        for key in ["HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HF_API_TOKEN"]:
            token = os.environ.get(key)
            if token:
                return token
        return HfFolder.get_token()
