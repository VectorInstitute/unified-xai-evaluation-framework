"""SHAP Global Feature Importance Ranking for TauBench Airline Dataset."""

import argparse

import pandas as pd


SHAP_PATH = "./data/taubench_airline_shap_per_run.csv"
OUTPUT_PATH = "./data/taubench_airline_shap_global_ranking.csv"

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Encode categorical labels in the dataset."
    )
    parser.add_argument(
        "--input", type=str, default=SHAP_PATH, help="Path to the input SHAP CSV file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_PATH,
        help="Path to save the global SHAP ranking CSV file.",
    )
    args = parser.parse_args()

    shap_df = pd.read_csv(args.input)

    X_cols = [
        "Intent Alignment",
        "Error Awareness & Recovery",
        "State Tracking Consistency",
        "Tool Correctness",
        "Tool Choice Accuracy",
        "Plan Adherence Metric",
    ]

    global_shap = (
        shap_df[X_cols]
        .abs()  # magnitude of contribution
        .mean()  # average across runs
        .sort_values(ascending=False)
        .reset_index()
    )

    global_shap.columns = ["attribute", "mean_abs_shap"]

    print(global_shap)
    global_shap.to_csv(args.output, index=False)
