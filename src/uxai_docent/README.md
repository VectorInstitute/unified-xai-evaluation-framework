# uxai_docent

Utilities to ingest HAL traces (AssistantBench and TAU-bench Airline) into Docent collections and to compute rubric evaluation metrics.


## Installation

Requires Python 3.12+.

- Install dependencies:
  - From [pyproject.toml](pyproject.toml) / [uv.lock](uv.lock) using uv:
    - Shell
      ```sh
      uv sync
      ```

## Docent API setup

- Set DOCENT_API_KEY in your environment:
  - Shell
    ```sh
    export DOCENT_API_KEY="your_key_here"
    ```

## Ingest HAL traces into Docent

AssistantBench:
- Shell
  ```sh
  python uxai_docent/ingest_docent_assistant.py \
    --trace-path <path_to_trace_file.json>
  ```

TAU-bench Airline:
- Shell
  ```sh
  python uxai_docent/ingest_docent_taubench.py \
    --trace-path <path_to_trace_file.json>
  ```

Key loader APIs:
- [`uxai_docent.ingest_docent_assistant.load_hal_weave_runs`](./ingest_docent_assistant.py)
- [`uxai_docent.ingest_docent_taubench.load_hal_weave_runs`](./ingest_docent_taubench.py)

## Compute rubric evaluation metrics

Input: CSV/XLSX with columns such as:
- task success/failure or task_outcome (values: success/failure)
- intent alignment - task (flag (match/no match))
- error awareness & recovery
- state tracking consistency
- tool correctness
- tool choice accuracy
- plan adherence metric

Run:
- Shell
  ```sh
  python uxai_docent/evaluation_metrics.py path/to/metrics_table.xlsx
  ```

APIs:
- [`uxai_docent.evaluation_metrics.load_table`](./evaluation_metrics.py)
- [`uxai_docent.evaluation_metrics.normalize_values`](./evaluation_metrics.py)
- [`uxai_docent.evaluation_metrics.contingency_table`](./evaluation_metrics.py)
- [`uxai_docent.evaluation_metrics.failure_mode_metrics`](./evaluation_metrics.py)
- [`uxai_docent.evaluation_metrics.reliability_metrics`](./evaluation_metrics.py)
- [`uxai_docent.evaluation_metrics.evaluate_rubric`](./evaluation_metrics.py)
