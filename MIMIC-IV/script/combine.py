#!/usr/bin/env python3
"""Combine model outputs from separate model scripts into summary files."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "result"
MODEL_OUT = RESULT
SUMMARY = RESULT / "summary"
SUMMARY.mkdir(exist_ok=True, parents=True)
CACHE = ROOT / "archive" / "cache"
CACHE.mkdir(exist_ok=True, parents=True)
os.environ.setdefault("MPLCONFIGDIR", str((CACHE / "mpl").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((CACHE / "xdg").resolve()))

import matplotlib.pyplot as plt


def main() -> None:
    perf_paths = [
        MODEL_OUT / "lm" / "perf.csv",
        MODEL_OUT / "gbm" / "perf.csv",
        MODEL_OUT / "rf" / "perf.csv",
        MODEL_OUT / "ebm" / "perf.csv",
        MODEL_OUT / "quant" / "perf.csv",
        MODEL_OUT / "nn" / "perf.csv",
    ]
    frames = [pd.read_csv(path) for path in perf_paths if path.exists()]
    if not frames:
        raise FileNotFoundError("No model performance files found. Run model scripts first.")

    performance = pd.concat(frames, ignore_index=True).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    performance.to_csv(SUMMARY / "perf.csv", index=False)

    plt.figure(figsize=(7, 4.5))
    plt.barh(performance["model"], performance["RMSE"], color="#356b8c")
    plt.gca().invert_yaxis()
    plt.xlabel("Test RMSE on log ICU LOS")
    plt.title("Model comparison on MIMIC-IV demo")
    plt.tight_layout()
    plt.savefig(SUMMARY / "perf.png", dpi=200)
    plt.close()

    print(f"Saved combined model performance to {SUMMARY / 'perf.csv'}")


if __name__ == "__main__":
    main()
