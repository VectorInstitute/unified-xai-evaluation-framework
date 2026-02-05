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

## Download HAL traces
- AssistantBench traces: [link](https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/assistantbench_assistantbench_browser_agent_gpt4120250414_1746225570_UPLOAD.zip?download=true)
- TAU-bench Airline traces: [link](https://huggingface.co/datasets/agent-evals/hal_traces/blob/main/taubench_airline_1743994890_UPLOAD.zip?download=true)

## Ingest HAL traces into Docent

AssistantBench:
- Shell
  ```sh
  python ingest_docent_assistant.py \
    --trace-path data/Traces/Assistantbench/assistantbench_assistantbench_browser_agent_gpt4120250414_1746225570_UPLOAD.json
  ```

TAU-bench Airline:
- Shell
  ```sh
  python ingest_docent_taubench.py \
    --trace-path data/Traces/Taubenchairline/taubench_airline_1743994890_UPLOAD.json
  ```

Key loader APIs:
- [`uxai_docent.ingest_docent_assistant.load_hal_weave_runs`](ingest_docent_assistant.py)
- [`uxai_docent.ingest_docent_taubench.load_hal_weave_runs`](ingest_docent_taubench.py)

Helpers:
- [`uxai_docent.ingest_docent_assistant.is_real_user_or_system_message`](ingest_docent_assistant.py)
- [`uxai_docent.ingest_docent_assistant.extract_assistant_payload`](ingest_docent_assistant.py)
- [`uxai_docent.ingest_docent_taubench.safe_content`](ingest_docent_taubench.py)


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
  python evaluation_metrics.py path/to/metrics_table.xlsx
  ```

APIs:
- [`uxai_docent.evaluation_metrics.load_table`](evaluation_metrics.py)
- [`uxai_docent.evaluation_metrics.normalize_values`](evaluation_metrics.py)
- [`uxai_docent.evaluation_metrics.contingency_table`](evaluation_metrics.py)
- [`uxai_docent.evaluation_metrics.failure_mode_metrics`](evaluation_metrics.py)
- [`uxai_docent.evaluation_metrics.reliability_metrics`](evaluation_metrics.py)
- [`uxai_docent.evaluation_metrics.evaluate_rubric`](evaluation_metrics.py)

## Behavioral attribution pipeline (encoding → SHAP → plots)

1) Encode rubric labels to numeric
- Script: [label_encoding.py](label_encoding.py)
- Shell
  ```sh
  python label_encoding.py \
    --input data/taubench_airline.xlsx \
    --output data/taubench_airline_encoded.xlsx
  ```

2) Fit logistic regression and compute per-run SHAP
- Script: [logistic_regression.py](logistic_regression.py)
- Shell
  ```sh
  python logistic_regression.py \
    --input data/taubench_airline_encoded.xlsx \
    --output data/taubench_airline_shap_per_run.csv
  ```

3) Aggregate global SHAP ranking
- Script: [shap_global.py](shap_global.py)
- Shell
  ```sh
  python shap_global.py \
    --input data/taubench_airline_shap_per_run.csv \
    --output data/taubench_airline_shap_global_ranking.csv
  ```

4) Visualize SHAP beeswarm
- Script: [shap_plot.py](shap_plot.py)
- Shell
  ```sh
  python shap_plot.py \
    --input data/taubench_airline_shap_per_run.csv \
    --output data/taubench_airline_shap_beeswarm.png
  ```

Notes:
- SHAP plotter uses a headless backend (Agg).
- Features used across the SHAP pipeline:
  - Intent Alignment
  - Error Awareness & Recovery
  - State Tracking Consistency
  - Tool Correctness
  - Tool Choice Accuracy
  - Plan Adherence Metric

## Data and prompts

- Data lives under [data/](data/). Example files:
  - [data/taubench_airline.xlsx](data/taubench_airline.xlsx)
  - [data/taubench_airline_encoded.xlsx](data/taubench_airline_encoded.xlsx)
  - [data/taubench_airline_shap_per_run.csv](data/taubench_airline_shap_per_run.csv)
  - [data/taubench_airline_shap_global_ranking.csv](data/taubench_airline_shap_global_ranking.csv)
  - [data/taubench_airline_shap_beeswarm.png](data/taubench_airline_shap_beeswarm.png)
  - [data/Traces/Assistantbench/assistantbench_assistantbench_browser_agent_gpt4120250414_1746225570_UPLOAD.json](data/Traces/Assistantbench/assistantbench_assistantbench_browser_agent_gpt4120250414_1746225570_UPLOAD.json)
  - [data/Traces/Taubenchairline/taubench_airline_1743994890_UPLOAD.json](data/Traces/Taubenchairline/taubench_airline_1743994890_UPLOAD.json)