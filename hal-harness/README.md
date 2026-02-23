# R2A2 GAIA Agent
Strict Closed-World Plan–Act–Reflect Reasoning Agent  
Built for the HAL Evaluation Framework

---

## 🧠 Overview

This repository contains a Strict Closed-World R2A2 Agent designed for the GAIA benchmark within the HAL evaluation framework.

The system integrates:

- Structured Plan–Act–Reflect reasoning
- Controlled and sandboxed tool usage
- Scratchpad memory management
- Closed-world capability enforcement
- GAIA dataset integration (HuggingFace)
- Responsible Reasoning (R2A2) evaluation metrics

The implementation emphasizes traceability, safety, determinism, and reproducibility.

---

## 🏗 Architecture

The agent follows a structured reasoning lifecycle:

PLAN → ACT → TOOL CALL → OBSERVE → REFLECT → REPLAN → FINAL ANSWER

Each reasoning step is logged into a trajectory for auditability.

The agent enforces:
- Bounded tool calls
- Bounded scratchpad growth
- Deterministic stopping
- Closed-world capability gating

---

## 📁 Project Structure

hal-harness/
│
├── agents/
│   └── r2a2_gaia/
│       ├── main.py
│       ├── requirements.txt
│       └── README.md
│
├── benchmarks/
│   └── gaia.py
│
├── evaluation.py

---

## 🔒 Closed-World Constraints

This agent strictly disallows:

- Internet browsing
- External API calls
- HTTP/HTTPS requests
- Wikipedia / Google references
- Image processing
- Audio processing
- External model calls

If a task requires unsupported capability, the agent returns:

UNSUPPORTED_TASK

This ensures reproducibility and prevents hidden external knowledge injection.

---

## 🛠 Tool System

The model may invoke tools using this exact format:

CALL: tool_name: argument

Example:
CALL: calculator: 12 * (5 + 3)

Supported Tools:

calculator  
Safe arithmetic evaluation using restricted AST parsing.

read_file  
Read local text, json, or log files.

load_table  
Load CSV/Excel files into a pandas DataFrame.

df_summary  
Return DataFrame overview.

df_head  
Return first N rows.

df_columns  
List column names.

df_shape  
Return table shape.

df_filter  
Safe pandas .query() execution with validation.

df_value_counts  
Frequency counts for a column.

df_sum  
Conditional column sum.

df_count  
Conditional row count.

All DataFrame operations are sandboxed with strict validation.

---

## 🧮 Safe Calculator

The calculator tool uses Python AST parsing with restricted operators.

Allowed:
+  -  *  /  //  %  **
Unary + and -

Blocked:
Imports
Function calls
Attribute access
OS interaction
Exec / eval
Dynamic code execution

---

## 📦 GAIA Benchmark Integration

File: benchmarks/gaia.py

The loader:

- Downloads GAIA dataset from HuggingFace:
  gaia-benchmark/GAIA
- Loads metadata from parquet
- Automatically retrieves attachments
- Supports GAIA levels 1, 2, 3
- Computes:
  - Overall accuracy
  - Per-level accuracy
  - Success/failure breakdown

### HuggingFace Authentication Required

Run:

huggingface-cli login

Or set environment variable:

export HF_TOKEN=your_token_here

---

## 🚀 Running the Agent

Example command:

hal-eval \
  --benchmark gaia \
  --agent_dir agents/r2a2_gaia \
  --agent_function main.run \
  --agent_name "DEBUG" \
  -A model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"

Optional overrides:

-A temperature=0.0
-A max_steps=6
-A max_new_tokens=512
-A scratchpad_max_chars=14000
-A auto_load_table=True
-A debug=True

---

## 🤖 Model Requirements

The agent uses HuggingFace Transformers.

Supported:
- LLaMA models
- Qwen models
- Any chat-template compatible model

Install required packages:

pip install torch transformers matplotlib

The agent automatically selects:
- bfloat16 if CUDA is available
- float32 otherwise

---

## 📋 Required Python Packages

Minimum required:

torch  
transformers  
pandas  
numpy  
datasets  
huggingface-hub  
pyarrow  
jsonlines  
matplotlib  

---

## 🔁 Agent Configuration

Configurable parameters:

model_id  
temperature  
top_p  
max_new_tokens  
max_steps  
max_tool_calls_per_step  
scratchpad_max_chars  
auto_load_table  
auto_read_text  
debug  

These prevent infinite loops and memory overflow.

---

## 📊 Responsible Reasoning Evaluation

File: evaluation.py

After GAIA execution completes, run:

python evaluation.py

This computes:

### Core Metrics

RC  — Task correctness  
TS  — Trace verbosity  
BM  — Baseline trust  
PI  — PII safety  
AL  — Agent lifecycle completeness  

---


| Metric   | Value     | Description |
|-----------|-----------|-------------|
| **RC**        | 0.054545 | Response Correctness (Task Success Rate) |
| **TS**        | 0.620000 | Trace Score (Reasoning Verbosity / Completeness) |
| **BM**        | 1.000000 | Baseline Model Trust |
| **PI**        | 0.993939 | PII Safety Score |
| **AL**        | 0.624242 | Agent Lifecycle Completeness |
| **RRI**       | 0.658545 | Responsible Reasoning Index |
| **CoT_RS**    | 0.667121 | Chain-of-Thought Responsibility Score |
| **RRC**       | 0.054545 | Raw Response Correctness |

---

### 📌 Interpretation

- The agent maintains **high safety (PI ≈ 0.99)**.
- Lifecycle execution is reasonably structured (**AL ≈ 0.62**).
- Overall responsibility (RRI ≈ 0.66) indicates stable governance behavior.
- Task correctness (RC ≈ 0.05) suggests performance improvements are needed on GAIA reasoning tasks.

### Derived Metrics

RRI — Responsible Reasoning Index  
Weighted combination of RC, TS, BM, PI, AL.

CoT_RS — Chain-of-Thought Responsibility Score  
Weighted reasoning quality metric.

AGR — Governance Aggregate  
System-level governance score.

CRRS — Composite Responsible Reasoning Score  
Final combined responsibility score.

---

## 📈 Evaluation Outputs

After running evaluation:

r2a2_metrics_output.csv  
r2a2_radar_plot.png  

The radar plot visualizes:

RC  
TS  
RRI  
CoT_RS  
RRC  
AGR  

---

## 📌 HAL Output Format

Agent returns:

{
  "task_id": {
    "answer": "...",
    "trajectory": [...]
  }
}

The trajectory includes:

Plan steps  
Tool calls  
Observations  
Reflections  
Replanning decisions  

---

## 🧪 Debug Mode

Enable debug logging:

-A debug=True

This prints:

Question preview  
Plan steps  
Tool execution logs  
Reflection decisions  
Final answer trace  

---

## ⚠️ Troubleshooting

Model not loading:
Ensure torch and transformers are installed.

GAIA dataset error:
Ensure huggingface-cli login is completed.

CUDA not detected:
Agent automatically falls back to CPU.

Infinite loops:
Reduce max_steps and max_tool_calls_per_step.

---

## 🎯 Summary

This R2A2 GAIA implementation provides:

Strict closed-world reasoning  
Tool-grounded structured execution  
Reflection-driven stopping  
Per-level GAIA scoring  
Responsible reasoning metrics  
Full audit trail logging  
Security-first design  

---