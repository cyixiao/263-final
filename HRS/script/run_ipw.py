#!/usr/bin/env python3
"""IPW-weighted analyses for physical activity and diabetes."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from interpret.glassbox import ExplainableBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import CATEGORICAL, RESULT, classification_metrics, get_xy, load_analytic, save_json


CACHE = Path(__file__).resolve().parents[1] / "archive" / "cache"
CACHE.mkdir(exist_ok=True, parents=True)
os.environ.setdefault("MPLCONFIGDIR", str((CACHE / "mpl").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((CACHE / "xdg").resolve()))

import matplotlib.pyplot as plt


FULL_FORMULA = (
    "diabetes ~ physical_act + age + bmi + C(female) + C(raceeth) + C(educ) + "
    "C(marital_status) + C(income) + C(smoking) + C(urban)"
)


def logit_or(df: pd.DataFrame, formula: str, weight_col: str | None, label: str) -> dict:
    if weight_col:
        model = smf.glm(formula, data=df, family=sm.families.Binomial(), freq_weights=df[weight_col])
        res = model.fit(cov_type="HC3")
    else:
        model = smf.logit(formula, data=df)
        res = model.fit(disp=False, maxiter=500)
    beta = res.params["physical_act"]
    ci = res.conf_int().loc["physical_act"]
    return {
        "model": label,
        "coef": float(beta),
        "OR": float(np.exp(beta)),
        "OR_low": float(np.exp(ci[0])),
        "OR_high": float(np.exp(ci[1])),
        "p_value": float(res.pvalues["physical_act"]),
    }


def main() -> None:
    df = load_analytic()
    out_dir = RESULT / "ipw"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        logit_or(df, "diabetes ~ physical_act", None, "unweighted_unadjusted_logit"),
        logit_or(df, FULL_FORMULA, None, "unweighted_adjusted_logit"),
        logit_or(df, "diabetes ~ physical_act", "ipw", "ipw_marginal_logit"),
        logit_or(df, FULL_FORMULA, "ipw", "ipw_adjusted_logit"),
    ]
    pd.DataFrame(rows).to_csv(out_dir / "physical_activity_or.csv", index=False)

    X, y = get_xy(df)
    feature_types = ["nominal" if c in CATEGORICAL else "continuous" for c in X.columns]
    ebm_unw = ExplainableBoostingClassifier(feature_types=feature_types, interactions=0, random_state=263)
    ebm_w = ExplainableBoostingClassifier(feature_types=feature_types, interactions=0, random_state=263)
    ebm_unw.fit(X, y)
    ebm_w.fit(X, y, sample_weight=df["ipw"])

    pred_unw = ebm_unw.predict_proba(X)[:, 1]
    pred_w = ebm_w.predict_proba(X)[:, 1]
    pd.DataFrame(
        [
            {"model": "unweighted_ebm", **classification_metrics(y, pred_unw)},
            {"model": "ipw_weighted_ebm", **classification_metrics(y, pred_w)},
        ]
    ).to_csv(out_dir / "ebm_perf.csv", index=False)
    pd.DataFrame({"hhidpn": df["hhidpn"], "y_true": y, "unweighted_ebm": pred_unw, "ipw_weighted_ebm": pred_w}).to_csv(
        out_dir / "ebm_pred.csv", index=False
    )

    def physical_scores(model: ExplainableBoostingClassifier) -> dict:
        exp = model.explain_global()
        idx = exp.data()["names"].index("physical_act")
        d = exp.data(idx)
        return {str(k): float(v) for k, v in zip(d["names"], d["scores"])}

    scores = pd.DataFrame(
        [
            {"model": "unweighted_ebm", **physical_scores(ebm_unw)},
            {"model": "ipw_weighted_ebm", **physical_scores(ebm_w)},
        ]
    )
    scores.to_csv(out_dir / "ebm_physical_activity_scores.csv", index=False)

    score_long = scores.melt(id_vars="model", var_name="physical_act", value_name="log_odds_contribution")
    plt.figure(figsize=(6, 4))
    for model, group in score_long.groupby("model"):
        plt.plot(group["physical_act"], group["log_odds_contribution"], marker="o", linewidth=2, label=model)
    plt.axhline(0, color="#555555", linestyle="--", linewidth=1)
    plt.xlabel("Physical activity")
    plt.ylabel("EBM log-odds contribution")
    plt.title("IPW changes EBM physical activity effect")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "ebm_physical_activity.png", dpi=220)
    plt.close()

    save_json(
        out_dir / "meta.json",
        {
            "note": "IPW analyses use the existing ipw column from the HRS analytic file.",
            "n": int(len(df)),
            "mean_ipw": float(df["ipw"].mean()),
            "max_ipw": float(df["ipw"].max()),
        },
    )
    print(f"Saved IPW analysis results to {out_dir}")


if __name__ == "__main__":
    main()
