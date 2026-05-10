#!/usr/bin/env python3
"""Combine HRS model outputs into summary files."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "result"
SUMMARY = RESULT / "summary"
SUMMARY.mkdir(exist_ok=True, parents=True)
CACHE = ROOT / "archive" / "cache"
CACHE.mkdir(exist_ok=True, parents=True)
os.environ.setdefault("MPLCONFIGDIR", str((CACHE / "mpl").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((CACHE / "xdg").resolve()))

import matplotlib.pyplot as plt


def main() -> None:
    perf_paths = [
        RESULT / "logit" / "perf.csv",
        RESULT / "rf" / "perf.csv",
        RESULT / "gbm" / "perf.csv",
        RESULT / "ebm" / "perf.csv",
        RESULT / "nn" / "perf.csv",
        RESULT / "spline" / "perf.csv",
    ]
    frames = [pd.read_csv(path) for path in perf_paths if path.exists()]
    if not frames:
        raise FileNotFoundError("No performance files found. Run model scripts first.")

    performance = pd.concat(frames, ignore_index=True).sort_values(["AUC", "PR_AUC"], ascending=False).reset_index(drop=True)
    performance.to_csv(SUMMARY / "perf.csv", index=False)

    plt.figure(figsize=(7, 4.5))
    plt.barh(performance["model"], performance["AUC"], color="#3c6e71")
    plt.gca().invert_yaxis()
    plt.xlim(0.5, max(1.0, performance["AUC"].max() + 0.02))
    plt.xlabel("Test AUC")
    plt.title("HRS diabetes model comparison")
    plt.tight_layout()
    plt.savefig(SUMMARY / "auc.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.barh(performance["model"], performance["Brier"], color="#6b7f82")
    plt.gca().invert_yaxis()
    plt.xlabel("Brier score")
    plt.title("Probability calibration error")
    plt.tight_layout()
    plt.savefig(SUMMARY / "brier.png", dpi=220)
    plt.close()

    print(f"Saved combined model performance to {SUMMARY / 'perf.csv'}")


if __name__ == "__main__":
    main()
