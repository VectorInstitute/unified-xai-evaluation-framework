# Traditional ML Text Classification with Explainability

This folder contains a reproducible experiment for text classification with explainability analyses. The main entry point is the notebook: [xai-experiment.ipynb](xai-experiment.ipynb).

## Overview

The notebook implements:
- Data preprocessing and stratified splits (by class and text length).
- Baseline checks against length confounding.
- Logistic Regression with TF-IDF and class weighting.
- Explainability:
  - SHAP global (beeswarm) and local (waterfall) explanations.
  - LIME local explanations with HTML export.
  - Token-level dependence plots for SHAP values.
  - Coefficient-based feature masking sensitivity tests.
  - Stability score via bootstrap Spearman rank correlation of explanations.
- CNN model with gradient-based saliency over embeddings for token-level importance.

## Data
The dataset used is a collection of job postings labeled as IT or Non-IT. It can be downloaded from https://www.kaggle.com/datasets/madhab/jobposts. The relevant file is to be renamed to `data_job_posts.csv` and placed in the current directory.

## Structure

Key steps inside [xai-experiment.ipynb](xai-experiment.ipynb):
- Get Data: Loads and cleans job postings; constructs “full_text”.
- Preprocessing: Length quantiles, stratified splits, loss class weights.
- Baselines: Length-only LR sanity checks.
- Model 1 (LR + TF-IDF): Training, evaluation, ROC-AUC.
- Explainability (LR):
  - SHAP beeswarm, waterfall for a single document.
  - SHAP dependence plots (continuous and binary presence).
  - LIME local explanation and HTML export.
  - Feature masking experiments (top IT/non-IT features).
  - Partial dependence via closed-form LR.
  - Holistic stability score via bootstrap + Spearman.
- Model 2 (CNN): Tokenization, padding, training, evaluation, gradient-based saliency.
