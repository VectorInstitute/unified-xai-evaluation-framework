"""Ingest HAL AssistantBench traces into Docent."""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

from docent import Docent
from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import parse_chat_message


TRACE_PATH = (
    "./data/Traces/Assistantbench/"
    "assistantbench_assistantbench_browser_agent_gpt4120250414_1746225570_UPLOAD.json"
)


def safe_str(val: Any) -> str:
    """Convert a value to string safely."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    return str(val)


def is_real_user_or_system_message(msg: Dict[str, Any]) -> bool:
    """
    Check if a message is a real user or system message.

    Keep ONLY real conversational messages.
    Drop browser UI dumps, screenshots, multimodal HUMAN messages.
    """
    role = msg.get("role")
    content = msg.get("content")

    if role not in {"system", "user"}:
        return False

    # Drop empty or non-string content
    if not isinstance(content, str):
        return False

    text = content.strip()
    if not text:
        return False

    # Drop browser state dumps explicitly
    banned_markers = [
        "[Current state starts here]",
        "Interactive elements",
        "Available tabs:",
        "Current url:",
        "Current step:",
        "chrome-error://",
        "Task history memory",
    ]

    return not any(marker in text for marker in banned_markers)


def extract_assistant_payload(msg: Dict[str, Any]) -> str:
    """
    Extract the assistant payload for AssistantBench.

    The REAL answer is inside tool_calls -> AgentOutput -> done.
    """
    parts = []

    if msg.get("tool_calls"):
        parts.append("[TOOL_CALLS]\n" + json.dumps(msg["tool_calls"], indent=2))

    if msg.get("invalid_tool_calls"):
        parts.append(
            "[INVALID_TOOL_CALLS]\n" + json.dumps(msg["invalid_tool_calls"], indent=2)
        )

    finish_reason = msg.get("response_metadata", {}).get("finish_reason")
    if finish_reason:
        parts.append(f"[FINISH_REASON] {finish_reason}")

    return "\n\n".join(parts).strip()


def load_hal_weave_runs(data: Dict[str, Any]) -> List[AgentRun]:  # noqa: PLR0912, PLR0915
    """
    Build ONE AgentRun per AssistantBench task (episode).

    Rules:
    - Group by weave_task_id
    - Keep ONLY real system + user intent messages
    - Use inputs.raw AIMessage as the authoritative assistant turn
    - Drop browser UI dumps & screenshots
    """
    config = data.get("config", {})
    benchmark_name = config.get("benchmark_name", "assistantbench")
    agent_name = config.get("agent_name", "unknown_agent")

    spans = data.get("raw_logging_results", [])

    spans_by_task: Dict[str, List[dict]] = defaultdict(list)
    for span in spans:
        task_id = span.get("weave_task_id")
        if task_id:
            spans_by_task[str(task_id)].append(span)

    agent_runs: List[AgentRun] = []

    for task_id, task_spans in spans_by_task.items():
        task_spans.sort(key=lambda s: s.get("started_at", ""))

        seen = set()
        messages = []

        for span in task_spans:
            inputs = span.get("inputs", {})

            raw_messages = inputs.get("messages", [])
            if isinstance(raw_messages, list):
                for msg in raw_messages:
                    if not isinstance(msg, dict):
                        continue

                    if not is_real_user_or_system_message(msg):
                        continue

                    role = msg["role"]
                    content = safe_str(msg["content"])
                    key = (role, content)

                    if key not in seen:
                        seen.add(key)
                        messages.append(
                            parse_chat_message({"role": role, "content": content})
                        )

            raw_ai = inputs.get("raw")
            if isinstance(raw_ai, dict) and raw_ai.get("_type") == "AIMessage":
                content = extract_assistant_payload(raw_ai)
                key = ("assistant", content)

                if content and key not in seen:
                    seen.add(key)
                    messages.append(
                        parse_chat_message({"role": "assistant", "content": content})
                    )

        if not messages:
            continue

        transcript = Transcript(
            messages=messages,
            metadata={
                "task_id": task_id,
                "benchmark": benchmark_name,
            },
        )

        agent_runs.append(
            AgentRun(
                transcripts=[transcript],
                metadata={
                    "task_id": task_id,
                    "benchmark": benchmark_name,
                    "agent_name": agent_name,
                    "total_cost": data.get("total_cost"),
                    "total_usage": data.get("total_usage"),
                },
            )
        )

    return agent_runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest HAL AssistantBench traces into Docent"
    )
    parser.add_argument(
        "--trace-path", type=str, default=TRACE_PATH, help="Path to the trace JSON file"
    )

    args = parser.parse_args()

    with open(args.trace_path, "r") as f:
        log = json.load(f)

    client = Docent(api_key=os.getenv("DOCENT_API_KEY"))

    collection_id = client.create_collection(
        name="HAL AssistantBench (Clean Episodes)",
        description=(
            "AssistantBench agent runs with UI dumps removed. "
            "One AgentRun per task. Tool calls and done() preserved."
        ),
    )

    print(f"Created collection: {collection_id}")

    agent_runs = load_hal_weave_runs(log)
    print(f"Ingesting {len(agent_runs)} agent runs...")

    client.add_agent_runs(collection_id, agent_runs)

    print("✅ Ingestion complete")
