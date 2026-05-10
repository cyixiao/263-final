#!/usr/bin/env python3
"""Prepare the HRS diabetes analytic dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ANALYTIC, DATA, RAW, save_json


RENAME = {
    "r10diabe": "diabetes",
    "physical_act_r10": "physical_act",
    "age_r10": "age",
    "educ_r10": "educ",
    "marital_status_r10": "marital_status",
    "bmi_r10": "bmi",
    "hshdinc_r10": "income",
    "smoking_r10": "smoking",
}


LABELS = {
    "diabetes": {"0": "No diabetes", "1": "Diabetes"},
    "physical_act": {"0": "No vigorous activity", "1": "Any vigorous activity"},
    "female": {"0": "Male", "1": "Female"},
    "raceeth": {"0": "Non-Hispanic White", "1": "Non-Hispanic Black", "2": "Hispanic", "3": "Other"},
    "educ": {
        "1": "Less than high school",
        "2": "High school graduate",
        "3": "Some college",
        "4": "College and above",
    },
    "marital_status": {"1": "Married", "2": "Separated/divorced", "3": "Widowed", "4": "Never married"},
    "income": {"0": "<25k", "1": "25k-49k", "2": "50k-99k", "3": "100k+"},
    "smoking": {"0": "No", "1": "Yes"},
    "urban_r10": {"0": "Suburban/rural", "1": "Urban"},
}


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(f"Missing raw data at {RAW}")

    df = pd.read_stata(RAW, convert_categoricals=False).rename(columns=RENAME)
    df = df.rename(columns={"urban_r10": "urban"})
    ordered_cols = [
        "hhidpn",
        "diabetes",
        "physical_act",
        "female",
        "age",
        "raceeth",
        "educ",
        "marital_status",
        "bmi",
        "income",
        "smoking",
        "urban",
        "pr_treat",
        "ipw",
    ]
    df = df[ordered_cols].copy()

    int_cols = ["diabetes", "physical_act", "female", "raceeth", "educ", "marital_status", "income", "smoking", "urban"]
    for col in int_cols:
        df[col] = df[col].astype(int)

    df.to_csv(ANALYTIC, index=False)

    table_rows = []
    for col in ["diabetes", "physical_act", "female", "raceeth", "educ", "marital_status", "income", "smoking", "urban"]:
        counts = df[col].value_counts(dropna=False).sort_index()
        for value, n in counts.items():
            table_rows.append(
                {
                    "variable": col,
                    "value": value,
                    "n": int(n),
                    "pct": float(n / len(df) * 100),
                }
            )
    pd.DataFrame(table_rows).to_csv(DATA / "table1_categorical.csv", index=False)
    df[["age", "bmi", "pr_treat", "ipw"]].describe().T.to_csv(DATA / "table1_continuous.csv")

    save_json(
        DATA / "meta.json",
        {
            "source": str(RAW),
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "outcome": "diabetes",
            "positive_class": "diabetes = 1",
            "prediction_note": "pr_treat and ipw are retained for causal/IPW analyses but excluded from default prediction features.",
            "labels": LABELS,
        },
    )
    print(f"Saved analytic HRS data to {ANALYTIC}")


if __name__ == "__main__":
    main()
