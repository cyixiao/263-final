#!/usr/bin/env python3
"""Calibration, ROC, and PR summaries for HRS diabetes classifiers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "result"
OUT = RESULT / "calib"
OUT.mkdir(exist_ok=True, parents=True)
CACHE = ROOT / "archive" / "cache"
CACHE.mkdir(exist_ok=True, parents=True)
os.environ.setdefault("MPLCONFIGDIR", str((CACHE / "mpl").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((CACHE / "xdg").resolve()))

import matplotlib.pyplot as plt


PRED_SPECS = [
    ("logit", RESULT / "logit" / "pred.csv", "logit"),
    ("elastic_net_logit", RESULT / "logit" / "pred.csv", "elastic_net_logit"),
    ("rf", RESULT / "rf" / "pred.csv", "rf"),
    ("gbm", RESULT / "gbm" / "pred.csv", "gbm"),
    ("ebm", RESULT / "ebm" / "pred.csv", "ebm"),
    ("mlp", RESULT / "nn" / "pred.csv", "mlp"),
    ("spline_logit", RESULT / "spline" / "pred.csv", "spline_logit"),
]


def logit_transform(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_stats(y: np.ndarray, p: np.ndarray) -> dict:
    x = logit_transform(p).reshape(-1, 1)
    cal = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    cal.fit(x, y)
    return {
        "calib_intercept": float(cal.intercept_[0]),
        "calib_slope": float(cal.coef_[0][0]),
        "mean_pred": float(np.mean(p)),
        "mean_obs": float(np.mean(y)),
        "Brier": float(brier_score_loss(y, p)),
        "AUC": float(roc_auc_score(y, p)),
        "PR_AUC": float(average_precision_score(y, p)),
    }


def calibration_bins(y: np.ndarray, p: np.ndarray, model: str, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y, "pred": p})
    df["bin"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")
    bins = (
        df.groupby("bin", observed=True)
        .agg(n=("y_true", "size"), pred_mean=("pred", "mean"), obs_rate=("y_true", "mean"))
        .reset_index(drop=True)
    )
    bins.insert(0, "model", model)
    return bins


def main() -> None:
    summaries = []
    bin_frames = []
    curves = {}

    for model, path, pred_col in PRED_SPECS:
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        if pred_col not in df.columns:
            continue
        y = df["y_true"].to_numpy(dtype=int)
        p = df[pred_col].to_numpy(dtype=float)
        summaries.append({"model": model, **calibration_stats(y, p)})
        bin_frames.append(calibration_bins(y, p, model))
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        curves[model] = {"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall}

    if not summaries:
        raise FileNotFoundError("No prediction files found. Run model scripts first.")

    summary = pd.DataFrame(summaries).sort_values(["Brier", "model"])
    bins = pd.concat(bin_frames, ignore_index=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    bins.to_csv(OUT / "bins.csv", index=False)

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1, label="ideal")
    for model, group in bins.groupby("model", sort=False):
        plt.plot(group["pred_mean"], group["obs_rate"], marker="o", linewidth=1.5, label=model)
    plt.xlabel("Mean predicted diabetes probability")
    plt.ylabel("Observed diabetes rate")
    plt.title("Calibration by prediction decile")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT / "calib.png", dpi=220)
    plt.close()

    plt.figure(figsize=(6, 5))
    for model, curve in curves.items():
        auc = summary.loc[summary["model"] == model, "AUC"].iloc[0]
        plt.plot(curve["fpr"], curve["tpr"], linewidth=1.5, label=f"{model} ({auc:.3f})")
    plt.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "roc.png", dpi=220)
    plt.close()

    plt.figure(figsize=(6, 5))
    for model, curve in curves.items():
        pr_auc = summary.loc[summary["model"] == model, "PR_AUC"].iloc[0]
        plt.plot(curve["recall"], curve["precision"], linewidth=1.5, label=f"{model} ({pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curves")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "pr.png", dpi=220)
    plt.close()

    print(f"Saved calibration results to {OUT}")


if __name__ == "__main__":
    main()
