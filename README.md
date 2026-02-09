# From Features to Actions: Explainability Across Static and Agentic AI Systems

### A Unified Experimental Framework for Static Attribution and Trajectory-Level Explainability

<p align="center" style="margin-top: -10px; margin-bottom: -10px;">
  <img src="docs/static/images/ComparisonOfMEP.png" width="420"/>
</p>
<br>

<p align="center">
  <b>📄 Paper:</b> <em>From Features to Actions: Explainability in Traditional and Agentic AI Systems</em>  
  &nbsp;|&nbsp;
<b>💻 Code:</b> <a href="https://github.com/VectorInstitute/unified-xai-evaluation-framework">GitHub</a>
  &nbsp;|&nbsp;
  <b>📊 Benchmarks:</b> TAU-bench Airline, AssistantBench, Job Postings
</p>

---

## 🧭 About

This repository contains the **reproducible experimental code** accompanying the paper:

> **From Features to Actions: Explainability in Traditional and Agentic AI Systems**

The project studies **how explainability requirements fundamentally change** when moving from:

* **Static prediction systems** (single input → single output), to
* **Agentic AI systems** (multi-step trajectories involving planning, tool use, and state updates)

We empirically compare:

* **Attribution-based explainability** (SHAP, LIME, saliency)
* **Trace-based, trajectory-level diagnostics** grounded in execution logs

across **both paradigms**, demonstrating that methods effective for static models **do not reliably diagnose agent failures**, motivating a shift toward **trajectory-level explainability**.

---

## ✨ Key Contributions

* 🔍 Empirical comparison of static vs. agentic explainability under a unified framework
* 📐 Formal distinction between **feature-level attribution** and **trajectory-level diagnostics**
* 🧪 Reproducible experiments spanning:

  * Traditional ML text classification
  * Tool-using LLM agents on real benchmarks
* 🧠 Introduction of **behavioral rubric–based failure analysis** for agentic systems
* 📦 End-to-end pipelines for ingestion, evaluation, and explainability analysis

---

## 🧠 Conceptual Overview

| Paradigm   | Unit of Explanation   | Primary Artifact           | Typical Question Answered      |
| ---------- | --------------------- | -------------------------- | ------------------------------ |
| Static ML  | Single prediction     | Feature attributions       | *Why this label?*              |
| Agentic AI | Multi-step trajectory | Execution traces + rubrics | *What failed, where, and why?* |

This repository operationalizes this distinction through **two complementary experimental tracks**, described below.

---

## 📦 Repository Structure

```
src/
│
├── uxai_docent/                 # Agentic explainability experiments
│
├── traditional_xai/             # Static ML explainability experiments
│   └── xai-experiment.ipynb
├── README.md                    # (this file)
```

---

## 🧪 Experiment 1: Traditional XAI for Static Text Classification (`traditional_xai/`)

This folder contains a **fully self-contained notebook experiment** for explainability in static prediction settings.

### Task

* Binary text classification: **IT vs. Non-IT job postings**
* Dataset: Kaggle Job Postings

### Models

* Logistic Regression + TF-IDF
* Text CNN baseline

### Explainability Methods

* SHAP (global & local)
* LIME (local, HTML export)
* Token-level SHAP dependence plots
* Gradient-based saliency (CNN)
* Feature masking sensitivity tests
* Bootstrap-based explanation stability (Spearman ρ)

---

## 🧪 Experiment 2: Agentic Explainability with Execution Traces (`uxai_docent/`)

This folder implements **trajectory-level explainability** for tool-using LLM agents evaluated on:

* **TAU-bench Airline**
* **AssistantBench**

### Core Capabilities

* Ingest full **HAL-Harness execution traces**
* Apply **Docent-based behavioral rubric evaluation**
* Quantify:

  * Failure-mode prevalence
  * Reliability correlates
* Bridge trace diagnostics with **SHAP over rubric features**

### Behavioral Rubrics

Each agent run is labeled using binary rubric flags:

* Intent Alignment
* Plan Adherence
* Tool Correctness
* Tool Choice Accuracy
* State Tracking Consistency
* Error Awareness & Recovery

These enable **per-run failure localization**, rather than post-hoc outcome explanation. This pipeline demonstrates that **trace-grounded diagnostics outperform attribution methods** for explaining agent failures.

---

### Key Finding

Attribution-based explanations are **stable and meaningful** in static settings, but **do not generalize** to explaining multi-step agent behavior.

---

## 🔬 Bridging the Paradigms

To directly compare explainability methods, we perform a **bridging experiment**:

1. Encode agent trajectories into **low-dimensional rubric features**
2. Train a logistic regression outcome predictor
3. Apply SHAP to rubric-level features

**Result:**
SHAP recovers sensible *global correlations*, but still fails to provide **trace-grounded, per-run diagnoses**, reinforcing the need for trajectory-level explanations.

---

## 📊 Data & Artifacts

* Agent traces (HAL-Harness JSON)
* Rubric evaluation tables (CSV / XLSX)
* SHAP outputs:

  * Per-run values
  * Global rankings
  * Beeswarm plots

All example paths are documented inside the respective subfolder READMEs.

---

## 📖 Citation

If you use this repository, please cite:

```bibtex
@article{featurestoactions2026,
  title   = {From Features to Actions: Explainability in Traditional and Agentic AI Systems},
  author = {Sindhuja Chaduvula, Jessee Ho, Kina Kim, Aravind Narayanan, Mahshid Alinoori, Muskan Garg, Dhanesh Ramachandram, Shaina Raza},
  journal = {arXiv preprint arXiv:2602.06841},
  year    = {2026}
}
```

---

## 📬 Contact

* Open a GitHub Issue for bugs or questions
* For research inquiries, contact the corresponding author listed in the paper (shaina.raza@vectorinstitute.ai)

---

## 🙏 Acknowledgments

This research was supported by the **Vector Institute for Artificial Intelligence** and funded in part by public and institutional partners.

---

**This repository is intended for researchers studying explainability, agent evaluation, and reliable AI systems.**
