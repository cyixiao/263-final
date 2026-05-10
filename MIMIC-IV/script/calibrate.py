#!/usr/bin/env python3
"""Summarize prediction calibration and interval coverage across models."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


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
    ("linear_regression", RESULT / "lm" / "pred.csv", "linear_regression", None, None),
    ("elastic_net", RESULT / "lm" / "pred.csv", "elastic_net", None, None),
    ("gbm", RESULT / "gbm" / "pred.csv", "gbm", None, None),
    ("rf", RESULT / "rf" / "pred.csv", "rf", None, None),
    ("ebm", RESULT / "ebm" / "pred.csv", "ebm", None, None),
    ("quant_median", RESULT / "quant" / "pred.csv", "q50", "q10", "q90"),
    ("mlp", RESULT / "nn" / "pred.csv", "mlp", None, None),
]


def _calibration_bins(y_true: np.ndarray, pred: np.ndarray, model: str, n_bins: int = 5) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true, "pred": pred})
    df["bin"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")
    bins = (
        df.groupby("bin", observed=True)
        .agg(n=("y_true", "size"), pred_mean=("pred", "mean"), obs_mean=("y_true", "mean"))
        .reset_index(drop=True)
    )
    bins.insert(0, "model", model)
    return bins


def main() -> None:
    summary_records = []
    bin_frames = []
    interval_records = []

    for model, path, pred_col, lo_col, hi_col in PRED_SPECS:
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        if pred_col not in df.columns:
            continue
        y_true = df["y_true"].to_numpy()
        pred = df[pred_col].to_numpy()

        reg = LinearRegression().fit(pred.reshape(-1, 1), y_true)
        summary_records.append(
            {
                "model": model,
                "calib_intercept": float(reg.intercept_),
                "calib_slope": float(reg.coef_[0]),
                "mean_pred": float(np.mean(pred)),
                "mean_obs": float(np.mean(y_true)),
                "mean_error": float(np.mean(pred - y_true)),
            }
        )
        bin_frames.append(_calibration_bins(y_true, pred, model))

        if lo_col and hi_col and lo_col in df.columns and hi_col in df.columns:
            interval_records.append(
                {
                    "model": model,
                    "interval": f"{lo_col}-{hi_col}",
                    "coverage": float(((df["y_true"] >= df[lo_col]) & (df["y_true"] <= df[hi_col])).mean()),
                    "mean_width": float((df[hi_col] - df[lo_col]).mean()),
                }
            )

    if not summary_records:
        raise FileNotFoundError("No prediction files found. Run model scripts first.")

    summary = pd.DataFrame(summary_records)
    bins = pd.concat(bin_frames, ignore_index=True)
    intervals = pd.DataFrame(interval_records)

    summary.to_csv(OUT / "summary.csv", index=False)
    bins.to_csv(OUT / "bins.csv", index=False)
    intervals.to_csv(OUT / "intervals.csv", index=False)

    plt.figure(figsize=(6.5, 5))
    lo = min(bins["pred_mean"].min(), bins["obs_mean"].min())
    hi = max(bins["pred_mean"].max(), bins["obs_mean"].max())
    plt.plot([lo, hi], [lo, hi], color="#555555", linewidth=1, linestyle="--", label="ideal")
    for model, group in bins.groupby("model", sort=False):
        plt.plot(group["pred_mean"], group["obs_mean"], marker="o", linewidth=1.5, label=model)
    plt.xlabel("Mean predicted log ICU LOS")
    plt.ylabel("Mean observed log ICU LOS")
    plt.title("Calibration by prediction quintile")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT / "calib.png", dpi=220)
    plt.close()

    if not intervals.empty:
        plt.figure(figsize=(5.5, 3.8))
        plt.bar(intervals["model"], intervals["coverage"], color="#5f8f62")
        plt.axhline(0.80, color="#555555", linewidth=1, linestyle="--")
        plt.ylim(0, 1)
        plt.ylabel("Empirical coverage")
        plt.title("80% prediction interval coverage")
        plt.tight_layout()
        plt.savefig(OUT / "intervals.png", dpi=220)
        plt.close()

    print(f"Saved calibration results to {OUT}")


if __name__ == "__main__":
    main()
