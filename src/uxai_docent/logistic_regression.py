"""Train a logistic regression model and compute SHAP values for feature attribution."""

import argparse

import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression


# Load encoded data
DATA_PATH = "./data/taubench_airline_encoded.xlsx"
OUTPUT_PATH = "./data/taubench_airline_shap_per_run.csv"


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Encode categorical labels in the dataset."
    )
    parser.add_argument(
        "--input", type=str, default=DATA_PATH, help="Path to the input Excel file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_PATH,
        help="Path to save the SHAP output CSV file.",
    )
    args = parser.parse_args()

    # Load data
    df = pd.read_excel(args.input)
    print(df.columns.tolist())

    # Features and label
    X_cols = [
        "Intent Alignment",
        "Error Awareness & Recovery",
        "State Tracking Consistency",
        "Tool Correctness",
        "Tool Choice Accuracy",
        "Plan Adherence Metric",
    ]

    X = df[X_cols]
    y = df["Task Success/Failure"]

    # Train tiny predictor
    model = LogisticRegression(penalty="l2", solver="liblinear", random_state=42)

    model.fit(X, y)

    # SHAP explainer
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)

    # Per-run SHAP attribution table
    shap_df = pd.DataFrame(shap_values, columns=X_cols)

    # Add metadata
    shap_df["task_id"] = df["Task ID "]
    shap_df["success"] = y.values

    # Save SHAP outputs
    shap_df.to_csv(args.output, index=False)

    print("SHAP attribution saved (per run)")
