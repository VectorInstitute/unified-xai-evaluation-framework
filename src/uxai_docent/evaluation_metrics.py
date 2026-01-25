"""Evaluation Metrics for Docent Rubrics."""

import sys
from pathlib import Path
from typing import Dict, Tuple, Union, cast

import pandas as pd


def load_table(path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV or XLSX file into a pandas DataFrame."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file type. Use CSV or XLSX.")

    df.columns = [c.strip().lower() for c in df.columns]
    return df


def normalize_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize string values in the DataFrame.

    Function does the following for each object-type column:
    - Converts to string
    - Strips leading/trailing whitespace
    - Converts to lowercase
    """
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def contingency_table(
    df: pd.DataFrame,
    outcome_col: str,
    rubric_col: str,
    flag_value: str = "no match",
) -> pd.DataFrame:
    """
    Build a contingency table for the given outcome and rubric columns.

    Builds a 2x2 contingency table:

                 Flag (no match)   No Flag (match)
    Failure
    Success
    """

    def count(outcome: str, flagged: bool) -> int:
        if flagged:
            return len(
                df[(df[outcome_col] == outcome) & (df[rubric_col] == flag_value)]
            )
        return len(df[(df[outcome_col] == outcome) & (df[rubric_col] != flag_value)])

    f_flag = count("failure", True)
    f_noflag = count("failure", False)
    s_flag = count("success", True)
    s_noflag = count("success", False)

    return pd.DataFrame(
        {
            "Flag (no match)": [f_flag, s_flag],
            "No Flag (match)": [f_noflag, s_noflag],
            "Total": [f_flag + f_noflag, s_flag + s_noflag],
        },
        index=["Failure", "Success"],
    )


def failure_mode_metrics(cont: pd.DataFrame) -> Dict[str, float]:
    """Compute failure-mode prevalence metrics from the contingency table."""
    f_flag = cast(int, cont.loc["Failure", "Flag (no match)"])
    f_total = cast(int, cont.loc["Failure", "Total"])
    s_flag = cast(int, cont.loc["Success", "Flag (no match)"])
    s_total = cast(int, cont.loc["Success", "Total"])

    p_f = f_flag / f_total if f_total > 0 else 0.0
    p_s = s_flag / s_total if s_total > 0 else 0.0

    return {
        "P(flag | task failure)": round(p_f, 3),
        "P(flag | task success)": round(p_s, 3),
        "Delta": round(p_f - p_s, 3),
        "Ratio": round(p_f / p_s, 3) if p_s > 0 else float("inf"),
    }


def reliability_metrics(cont: pd.DataFrame) -> Dict[str, float]:
    """Compute reliability-correlate metrics from the contingency table."""
    s_flag = cast(int, cont.loc["Success", "Flag (no match)"])
    f_flag = cast(int, cont.loc["Failure", "Flag (no match)"])
    s_noflag = cast(int, cont.loc["Success", "No Flag (match)"])
    f_noflag = cast(int, cont.loc["Failure", "No Flag (match)"])

    total_flag = s_flag + f_flag
    total_noflag = s_noflag + f_noflag

    p_s_flag = s_flag / total_flag if total_flag > 0 else 0.0
    p_s_noflag = s_noflag / total_noflag if total_noflag > 0 else 0.0

    return {
        "P(task success | flag)": round(p_s_flag, 3),
        "P(task success | no flag)": round(p_s_noflag, 3),
        "Delta": round(p_s_flag - p_s_noflag, 3),
        "RR": round(p_s_flag / p_s_noflag, 3) if p_s_noflag > 0 else float("inf"),
    }


def evaluate_rubric(
    df: pd.DataFrame, rubric: str
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """Evaluate a single rubric column against task outcomes."""
    if "task success/failure" in df.columns:
        outcome_col = "task success/failure"
    elif "task_outcome" in df.columns:
        outcome_col = "task_outcome"
    else:
        raise KeyError("Task outcome column not found.")

    cont = contingency_table(df, outcome_col, rubric)
    fm = failure_mode_metrics(cont)
    rel = reliability_metrics(cont)

    return cont, fm, rel


def main(path: str) -> None:
    """Load data and evaluate metrics for each rubric."""
    df = load_table(path)
    df = normalize_values(df)

    rubrics = [
        "intent alignment - task (flag (match/no match))",
        "error awareness & recovery",
        "state tracking consistency",
        "tool correctness",
        "tool choice accuracy",
        "plan adherence metric",
    ]

    print("\n================ METRICS ================\n")

    for rubric in rubrics:
        if rubric not in df.columns:
            print(f"[SKIP] Column not found: {rubric}")
            continue

        print(f"\n--- Rubric: {rubric.upper()} ---\n")

        cont, fm, rel = evaluate_rubric(df, rubric)

        print("Contingency Table:")
        print(cont, "\n")

        print("Failure-Mode Prevalence (Table A3):")
        for k, v in fm.items():
            print(f"  {k}: {v}")
        print()

        print("Reliability Correlates (Table A4):")
        for k, v in rel.items():
            print(f"  {k}: {v}")

        print("\n" + "-" * 60)

    print("\nDone.\n")


if __name__ == "__main__":
    """Entry point for command-line execution."""
    if len(sys.argv) != 2:
        print("Usage: python evaluation_metrics.py <input.csv|input.xlsx>")
        sys.exit(1)

    main(sys.argv[1])
