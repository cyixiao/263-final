#!/usr/bin/env python3
"""Restricted cubic spline logistic analysis for HRS diabetes risk."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrix
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import RESULT, classification_metrics, load_analytic, save_json


CACHE = Path(__file__).resolve().parents[1] / "archive" / "cache"
CACHE.mkdir(exist_ok=True, parents=True)
os.environ.setdefault("MPLCONFIGDIR", str((CACHE / "mpl").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((CACHE / "xdg").resolve()))

import matplotlib.pyplot as plt


CAT_COLS = ["physical_act", "female", "raceeth", "educ", "marital_status", "income", "smoking", "urban"]


def make_design(df: pd.DataFrame, design_info: dict | None = None) -> tuple[pd.DataFrame, list[str], list[str], dict]:
    if design_info is None:
        age_spline = dmatrix("cr(age, df=4) - 1", df, return_type="dataframe")
        bmi_spline = dmatrix("cr(bmi, df=4) - 1", df, return_type="dataframe")
        design_info = {"age": age_spline.design_info, "bmi": bmi_spline.design_info}
    else:
        age_spline = pd.DataFrame(build_design_matrices([design_info["age"]], df)[0])
        bmi_spline = pd.DataFrame(build_design_matrices([design_info["bmi"]], df)[0])
    age_spline.columns = [f"age_spline_{i + 1}" for i in range(age_spline.shape[1])]
    bmi_spline.columns = [f"bmi_spline_{i + 1}" for i in range(bmi_spline.shape[1])]
    age_spline.index = df.index
    bmi_spline.index = df.index
    cats = pd.get_dummies(df[CAT_COLS].astype("category"), columns=CAT_COLS, drop_first=True, dtype=float)
    X = pd.concat([age_spline, bmi_spline, cats], axis=1)
    X = sm.add_constant(X, has_constant="add")
    return X, list(age_spline.columns), list(bmi_spline.columns), design_info


def odds_table(result, variable_names: list[str]) -> pd.DataFrame:
    rows = []
    for var in variable_names:
        if var not in result.params.index:
            continue
        beta = result.params[var]
        ci = result.conf_int().loc[var]
        rows.append(
            {
                "variable": var,
                "coef": float(beta),
                "OR": float(np.exp(beta)),
                "OR_low": float(np.exp(ci[0])),
                "OR_high": float(np.exp(ci[1])),
                "p_value": float(result.pvalues[var]),
            }
        )
    return pd.DataFrame(rows)


def plot_adjusted_curve(df: pd.DataFrame, result, design_info: dict, variable: str, out_path: Path) -> None:
    base = df.copy()
    grid = np.linspace(df[variable].quantile(0.01), df[variable].quantile(0.99), 120)
    ref = df.median(numeric_only=True).to_dict()
    fixed = pd.DataFrame({variable: grid})
    fixed["age"] = ref["age"]
    fixed["bmi"] = ref["bmi"]
    fixed[variable] = grid
    for col in CAT_COLS:
        fixed[col] = int(df[col].mode().iloc[0])
    X_grid, _, _, _ = make_design(fixed, design_info=design_info)
    X_grid = X_grid.reindex(columns=result.params.index, fill_value=0)
    pred = result.get_prediction(X_grid).summary_frame()
    plt.figure(figsize=(6, 4))
    plt.plot(grid, pred["mean"], color="#3c6e71", linewidth=2)
    plt.fill_between(grid, pred["mean_ci_lower"], pred["mean_ci_upper"], color="#3c6e71", alpha=0.18, linewidth=0)
    plt.xlabel(variable.upper())
    plt.ylabel("Adjusted diabetes probability")
    plt.title(f"Restricted cubic spline: {variable}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    df = load_analytic()
    out_dir = RESULT / "spline"
    out_dir.mkdir(parents=True, exist_ok=True)

    X, age_terms, bmi_terms, design_info = make_design(df)
    y = df["diabetes"].astype(int)
    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()
    proba = result.predict(X)

    pd.DataFrame([{"model": "spline_logit", **classification_metrics(y, proba)}]).to_csv(out_dir / "perf.csv", index=False)
    pd.DataFrame({"hhidpn": df["hhidpn"], "y_true": y, "spline_logit": proba}).to_csv(out_dir / "pred.csv", index=False)
    odds_table(result, [c for c in X.columns if c not in ["const", *age_terms, *bmi_terms]]).to_csv(
        out_dir / "or_table.csv", index=False
    )
    test_rows = []
    for term_name, terms in [("age_spline", age_terms), ("bmi_spline", bmi_terms)]:
        reduced = sm.GLM(y, X.drop(columns=terms), family=sm.families.Binomial()).fit()
        lr_stat = 2 * (result.llf - reduced.llf)
        df_diff = len(terms)
        test_rows.append(
            {
                "term": term_name,
                "lr_chi2": float(lr_stat),
                "df": int(df_diff),
                "p_value": float(chi2.sf(lr_stat, df_diff)),
            }
        )
    pd.DataFrame(test_rows).to_csv(out_dir / "spline_tests.csv", index=False)

    plot_adjusted_curve(df, result, design_info, "age", out_dir / "age_curve.png")
    plot_adjusted_curve(df, result, design_info, "bmi", out_dir / "bmi_curve.png")
    save_json(
        out_dir / "meta.json",
        {
            "model": "statsmodels GLM binomial with restricted cubic splines",
            "n": int(len(df)),
            "age_spline_df": 4,
            "bmi_spline_df": 4,
            "test": "likelihood ratio tests comparing full spline model with reduced models",
        },
    )
    print(f"Saved spline logistic results to {out_dir}")


if __name__ == "__main__":
    main()
