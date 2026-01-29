"""SHAP Beeswarm Plot for TauBench Airline Dataset."""

import matplotlib as mpl
import pandas as pd
import shap


mpl.use("Agg")
import argparse

import matplotlib.pyplot as plt


# Load per-run SHAP values
SHAP_PATH = "./data/taubench_airline_shap_per_run.csv"
OUTPUT_PATH = "./data/taubench_airline_shap_beeswarm.png"

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
        help="Path to save the encoded Excel file.",
    )
    args = parser.parse_args()

    # Load per-run SHAP values
    shap_df = pd.read_csv(args.input)

    X_cols = [
        "Intent Alignment",
        "Error Awareness & Recovery",
        "State Tracking Consistency",
        "Tool Correctness",
        "Tool Choice Accuracy",
        "Plan Adherence Metric",
    ]

    shap_values = shap_df[X_cols].values
    feature_values = shap_df[X_cols]

    # SHAP beeswarm plot
    plt.figure(figsize=(7, 4))
    shap.summary_plot(shap_values, feature_values, plot_type="dot", show=False)

    plt.title("Global SHAP Summary (Behavioral Dimensions)")
    plt.tight_layout()

    plt.savefig(args.output)
    plt.close()

    print(f"SHAP beeswarm saved to: {args.output}")
