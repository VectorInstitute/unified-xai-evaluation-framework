"""Label Encoding Script for Categorical Data."""

import argparse

import pandas as pd


# Default Paths
INPUT_PATH = "./data/taubench_airline.xlsx"
OUTPUT_PATH = "./data/taubench_airline_encoded.xlsx"


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Encode categorical labels in the dataset."
    )
    parser.add_argument(
        "--input", type=str, default=INPUT_PATH, help="Path to the input Excel file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_PATH,
        help="Path to save the encoded Excel file.",
    )
    args = parser.parse_args()

    # Load data
    df = pd.read_excel(args.input)

    # Encode mappings
    match_map = {"match": 1, "no match": 0}

    outcome_map = {"success": 1, "failure": 0}

    # Apply encodings
    df["Task Success/Failure"] = df["Task Success/Failure"].map(outcome_map)

    metric_columns = [
        "Intent Alignment - Task (Flag (match/no match))",
        "Error Awareness & Recovery",
        "State Tracking Consistency",
        "Tool Correctness",
        "Tool Choice Accuracy",
        "Plan Adherence Metric",
    ]

    for col in metric_columns:
        df[col] = df[col].map(match_map)

    # Save encoded file
    df.to_excel(args.output, index=False)
    print("Encoding complete. Saved to:", OUTPUT_PATH)
