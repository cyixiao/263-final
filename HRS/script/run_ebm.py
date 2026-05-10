#!/usr/bin/env python3
"""Run EBM classifier and save explanation plots for HRS diabetes prediction."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "archive" / "cache"
CACHE.mkdir(exist_ok=True, parents=True)
os.environ.setdefault("MPLCONFIGDIR", str((CACHE / "mpl").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((CACHE / "xdg").resolve()))

import matplotlib.backends.backend_pdf as backend_pdf
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import CATEGORICAL, RESULT, classification_metrics, get_xy, load_analytic, save_json

import matplotlib.pyplot as plt


def draw_term(ax, names: list, scores: np.ndarray, lower: np.ndarray | None, upper: np.ndarray | None, title: str) -> None:
    x = np.arange(len(scores))
    if len(scores) <= 12:
        ax.bar(x, scores, color="#3c6e71")
        ax.set_xticks(x)
        labels = names[: len(scores)] if names else [str(i) for i in x]
        ax.set_xticklabels(labels, rotation=35, ha="right")
    else:
        ax.plot(x, scores, color="#3c6e71", linewidth=2)
        if lower is not None and upper is not None and len(lower) == len(scores) and len(upper) == len(scores):
            ax.fill_between(x, lower, upper, color="#3c6e71", alpha=0.18, linewidth=0)
        tick_idx = np.linspace(0, len(scores) - 1, min(6, len(scores))).astype(int)
        ax.set_xticks(tick_idx)
        if names:
            ax.set_xticklabels([names[min(i, len(names) - 1)] for i in tick_idx])
    ax.axhline(0, color="#555555", linewidth=0.9)
    ax.set_title(title)
    ax.set_ylabel("Log-odds contribution")
    ax.tick_params(axis="both", labelsize=8)


def plot_term(names: list, scores: np.ndarray, lower: np.ndarray | None, upper: np.ndarray | None, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    draw_term(ax, names, scores, lower, upper, title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_ebm_plots(ebm: ExplainableBoostingClassifier, out_dir: Path) -> None:
    figs_dir = out_dir / "figs"
    if figs_dir.exists():
        shutil.rmtree(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)

    global_exp = ebm.explain_global()
    data = global_exp.data()
    names = data["names"]
    scores = data["scores"]
    importance = pd.DataFrame({"term": names, "importance": scores}).sort_values("importance", ascending=False)
    importance.to_csv(out_dir / "importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    top = importance.head(15).sort_values("importance")
    ax.barh(top["term"], top["importance"], color="#3c6e71")
    ax.set_xlabel("Mean absolute score")
    ax.set_title("EBM term importance")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_global.pdf")
    plt.close(fig)

    term_plot_data = []
    for i, term in enumerate(names):
        term_data = global_exp.data(i)
        term_names = [str(v) for v in term_data.get("names", [])]
        term_scores = np.asarray(term_data.get("scores", []), dtype=float)
        lower = term_data.get("lower_bounds")
        upper = term_data.get("upper_bounds")
        lower = np.asarray(lower, dtype=float) if lower is not None else None
        upper = np.asarray(upper, dtype=float) if upper is not None else None
        clean = "".join(ch if ch.isalnum() else "_" for ch in term)[:60].strip("_")
        path = figs_dir / f"{i + 1:02d}_{clean}.pdf"
        plot_term(term_names, term_scores, lower, upper, term, path)
        term_plot_data.append((term_names, term_scores, lower, upper, term))

    combined = out_dir / "fig_all.pdf"
    with backend_pdf.PdfPages(combined) as pdf:
        per_page = 6
        for start in range(0, len(term_plot_data), per_page):
            subset = term_plot_data[start : start + per_page]
            fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
            for ax, plot_args in zip(axes.ravel(), subset):
                draw_term(ax, *plot_args)
            for ax in axes.ravel()[len(subset) :]:
                ax.set_axis_off()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    df = load_analytic()
    X, y = get_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=263, stratify=y)

    feature_types = ["nominal" if c in CATEGORICAL else "continuous" for c in X.columns]
    model = ExplainableBoostingClassifier(
        feature_types=feature_types,
        interactions=0,
        learning_rate=0.03,
        max_bins=256,
        random_state=263,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    out_dir = RESULT / "ebm"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": "ebm", **classification_metrics(y_test, proba)}]).to_csv(out_dir / "perf.csv", index=False)
    pd.DataFrame({"y_true": y_test.to_numpy(), "ebm": proba}, index=y_test.index).to_csv(out_dir / "pred.csv", index=True)
    save_ebm_plots(model, out_dir)
    save_json(
        out_dir / "meta.json",
        {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "interactions": 0,
            "feature_types": dict(zip(X.columns, feature_types)),
        },
    )
    print(f"Saved EBM results to {out_dir}")


if __name__ == "__main__":
    main()
