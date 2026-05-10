#!/usr/bin/env python3
"""Subgroup and interaction analyses for physical activity and diabetes."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import RESULT, load_analytic, save_json


CACHE = Path(__file__).resolve().parents[1] / "archive" / "cache"
CACHE.mkdir(exist_ok=True, parents=True)
os.environ.setdefault("MPLCONFIGDIR", str((CACHE / "mpl").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((CACHE / "xdg").resolve()))

import matplotlib.pyplot as plt


BASE_FORMULA = (
    "diabetes ~ physical_act + age + bmi + C(female) + C(raceeth) + C(educ) + "
    "C(marital_status) + C(income) + C(smoking) + C(urban)"
)


def fit_or(df: pd.DataFrame, formula: str = BASE_FORMULA) -> tuple[float, float, float, float]:
    res = smf.logit(formula, data=df).fit(disp=False, maxiter=500)
    beta = res.params["physical_act"]
    ci = res.conf_int().loc["physical_act"]
    return float(np.exp(beta)), float(np.exp(ci[0])), float(np.exp(ci[1])), float(res.pvalues["physical_act"])


def main() -> None:
    df = load_analytic()
    out_dir = RESULT / "subgroup"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["age_group"] = np.where(df["age"] < 65, "51-64", "65+")
    df["bmi_group"] = pd.cut(
        df["bmi"],
        bins=[0, 25, 30, np.inf],
        labels=["normal/underweight", "overweight", "obese"],
        right=False,
    )
    df["income_group"] = np.where(df["income"] <= 1, "low/mid income", "higher income")

    subgroup_specs = {
        "female": "female",
        "age_group": "age_group",
        "bmi_group": "bmi_group",
        "income_group": "income_group",
        "raceeth": "raceeth",
    }
    rows = []
    overall = fit_or(df)
    rows.append({"subgroup": "overall", "level": "all", "OR": overall[0], "OR_low": overall[1], "OR_high": overall[2], "p": overall[3], "n": len(df)})
    for subgroup, col in subgroup_specs.items():
        for level, group in df.groupby(col, observed=True):
            if group["diabetes"].nunique() < 2 or group["physical_act"].nunique() < 2 or len(group) < 200:
                continue
            try:
                or_val, lo, hi, p = fit_or(group)
                rows.append(
                    {
                        "subgroup": subgroup,
                        "level": str(level),
                        "OR": or_val,
                        "OR_low": lo,
                        "OR_high": hi,
                        "p": p,
                        "n": int(len(group)),
                    }
                )
            except Exception as exc:
                rows.append({"subgroup": subgroup, "level": str(level), "error": str(exc), "n": int(len(group))})

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "physical_activity_or_by_subgroup.csv", index=False)

    interaction_rows = []
    for term in ["C(female)", "age", "bmi", "C(income)", "C(raceeth)"]:
        formula = BASE_FORMULA.replace("physical_act", f"physical_act * {term}", 1)
        try:
            res = smf.logit(formula, data=df).fit(disp=False, maxiter=500)
            for name, p in res.pvalues.items():
                if name.startswith("physical_act:") or name.startswith(f"{term}:physical_act"):
                    interaction_rows.append({"interaction": name, "coef": float(res.params[name]), "OR": float(np.exp(res.params[name])), "p": float(p)})
        except Exception as exc:
            interaction_rows.append({"interaction": f"physical_act x {term}", "error": str(exc)})
    pd.DataFrame(interaction_rows).to_csv(out_dir / "interaction_tests.csv", index=False)

    plot_df = table[table["subgroup"].ne("overall") & table["OR"].notna()].copy()
    plot_df["label"] = plot_df["subgroup"] + ": " + plot_df["level"]
    plot_df = plot_df.sort_values("OR")
    plt.figure(figsize=(7, max(4, 0.35 * len(plot_df))))
    y_pos = np.arange(len(plot_df))
    plt.errorbar(
        plot_df["OR"],
        y_pos,
        xerr=[plot_df["OR"] - plot_df["OR_low"], plot_df["OR_high"] - plot_df["OR"]],
        fmt="o",
        color="#3c6e71",
        ecolor="#6b7f82",
        capsize=3,
    )
    plt.axvline(1, color="#555555", linestyle="--", linewidth=1)
    plt.yticks(y_pos, plot_df["label"])
    plt.xlabel("Adjusted OR for physical activity")
    plt.title("Physical activity association by subgroup")
    plt.tight_layout()
    plt.savefig(out_dir / "subgroup_or.png", dpi=220)
    plt.close()

    save_json(out_dir / "meta.json", {"model": "adjusted logistic regression subgroup and interaction analyses"})
    print(f"Saved subgroup analysis results to {out_dir}")


if __name__ == "__main__":
    main()
