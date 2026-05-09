#!/usr/bin/env python3
"""Export editable matplotlib EBM explanation plots.

This is the "manual" plot style: every figure is drawn from EBM explanation
data with matplotlib, so fonts, colors, labels, and sizing are easy to adjust
later. The global term-importance overview is saved as its own PDF and is not
included in the combined variable-level PDF.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULT = ROOT / "result"
ANALYTIC = DATA / "analytic.csv"
RESULT.mkdir(exist_ok=True, parents=True)
CACHE = ROOT / "archive" / "cache"
CACHE.mkdir(exist_ok=True, parents=True)
os.environ.setdefault("MPLCONFIGDIR", str((CACHE / "mpl").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((CACHE / "xdg").resolve()))

import matplotlib.pyplot as plt

DROP_COLS = {
    "subject_id",
    "hadm_id",
    "stay_id",
    "intime",
    "outtime",
    "anchor_year",
    "anchor_year_group",
    "dod",
    "admittime",
    "dischtime",
    "deathtime",
    "edregtime",
    "edouttime",
    "los",
    "log_los",
    "anchor_age",
    "race",
    "last_careunit",
    "admit_provider_id",
    "discharge_location",
    "hospital_expire_flag",
}


def safe_name(value: object, limit: int = 90) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)[:limit]


def compact_labels(labels, max_len: int = 28) -> list[str]:
    out = []
    for label in labels:
        text = str(label)
        if len(text) > max_len:
            text = text[: max_len - 3] + "..."
        out.append(text)
    return out


def build_ebm_input(missing_threshold: float = 0.80) -> tuple[pd.DataFrame, pd.Series]:
    if not ANALYTIC.exists():
        raise FileNotFoundError(
            f"Missing {ANALYTIC}. Run script/prep_data.py first."
        )
    df = pd.read_csv(ANALYTIC)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    missing = df[feature_cols].isna().mean()
    feature_cols = [c for c in feature_cols if missing[c] <= missing_threshold]

    X = df[feature_cols].copy()
    y = df["log_los"].copy()
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")
    return X, y


def fit_ebm(X: pd.DataFrame, y: pd.Series) -> ExplainableBoostingRegressor:
    ebm = ExplainableBoostingRegressor(
        random_state=263,
        interactions=3,
        max_bins=32,
        outer_bags=4,
        n_jobs=1,
    )
    ebm.fit(X, y)
    return ebm


def draw_overview(ax: plt.Axes, explanation) -> None:
    data = explanation.data()
    names = np.asarray(data["names"], dtype=object)
    scores = np.asarray(data["scores"], dtype=float)
    order = np.argsort(scores)
    names = names[order]
    scores = scores[order]
    ax.barh(np.arange(len(scores)), scores, color="#356b8c")
    ax.set_yticks(np.arange(len(scores)))
    ax.set_yticklabels(compact_labels(names, 34), fontsize=6)
    ax.set_xlabel("Mean absolute contribution")
    ax.set_title("EBM global term importance", fontsize=11)
    ax.grid(axis="x", alpha=0.25)


def draw_term(ax: plt.Axes, data: dict, title: str) -> None:
    if data.get("type") == "interaction":
        scores = np.asarray(data["scores"], dtype=float)
        im = ax.imshow(scores.T, origin="lower", aspect="auto", cmap="RdBu_r")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(compact_labels([title.split(" & ")[0]], 24)[0])
        ax.set_ylabel(compact_labels([title.split(" & ")[-1]], 24)[0])

        left = data.get("left_names", [])
        right = data.get("right_names", [])
        if left:
            ticks = np.linspace(0, len(left) - 2, min(5, max(1, len(left) - 1))).astype(int)
            ax.set_xticks(ticks)
            ax.set_xticklabels(compact_labels([left[t] for t in ticks], 10), rotation=45, ha="right", fontsize=6)
        if right:
            ticks = np.linspace(0, len(right) - 2, min(5, max(1, len(right) - 1))).astype(int)
            ax.set_yticks(ticks)
            ax.set_yticklabels(compact_labels([right[t] for t in ticks], 10), fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return

    names = list(data.get("names", []))
    scores = np.asarray(data.get("scores", []), dtype=float)
    lower = np.asarray(data.get("lower_bounds", scores), dtype=float)
    upper = np.asarray(data.get("upper_bounds", scores), dtype=float)

    if len(names) == len(scores) + 1 and all(isinstance(x, (int, float, np.number)) for x in names):
        edges = np.asarray(names, dtype=float)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, scores, color="#2f6f4e", linewidth=1.8)
        ax.fill_between(centers, lower, upper, color="#2f6f4e", alpha=0.18)
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_xlabel(title)
        ax.set_ylabel("Contribution to log ICU LOS")
        ax.grid(alpha=0.25)
    else:
        labels = compact_labels(names, 22)
        yerr = np.vstack([scores - lower, upper - scores])
        ax.bar(np.arange(len(scores)), scores, yerr=yerr, color="#8c5a35", alpha=0.88, capsize=2)
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Contribution to log ICU LOS")
        ax.grid(axis="y", alpha=0.25)
    ax.set_title(title, fontsize=9)


def save_single_term(explanation, index: int, title: str, pdf_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    draw_term(ax, explanation.data(index), title)
    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def export_manual_plots(
    ebm: ExplainableBoostingRegressor,
    separate_dir: Path,
    combined_pdf: Path,
    overview_pdf: Path,
    n_cols: int = 3,
) -> None:
    separate_dir.mkdir(parents=True, exist_ok=True)
    explanation = ebm.explain_global()
    term_names = list(ebm.term_names_)

    fig, ax = plt.subplots(figsize=(8.5, 11))
    draw_overview(ax, explanation)
    fig.tight_layout()
    fig.savefig(overview_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    for i, term_name in enumerate(term_names):
        stem = f"{i + 1:02d}_{safe_name(term_name)}"
        save_single_term(explanation, i, str(term_name), separate_dir / f"{stem}.pdf")

    n_rows = (len(term_names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = np.asarray(axes).flatten()
    for i, term_name in enumerate(term_names):
        draw_term(axes[i], explanation.data(i), str(term_name))
    for j in range(len(term_names), len(axes)):
        axes[j].axis("off")
    plt.subplots_adjust(wspace=0.22, hspace=0.32)
    plt.tight_layout(pad=0.6)
    fig.savefig(combined_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--separate-dir",
        type=Path,
        default=RESULT / "ebm" / "figs",
    )
    parser.add_argument(
        "--combined-pdf",
        type=Path,
        default=RESULT / "ebm" / "fig_all.pdf",
    )
    parser.add_argument(
        "--overview-pdf",
        type=Path,
        default=RESULT / "ebm" / "fig_global.pdf",
    )
    args = parser.parse_args()

    X, y = build_ebm_input()
    ebm = fit_ebm(X, y)
    export_manual_plots(ebm, args.separate_dir, args.combined_pdf, args.overview_pdf)
    print(f"Saved editable manual PDFs to: {args.separate_dir}")
    print(f"Saved combined variable PDF to: {args.combined_pdf}")
    print(f"Saved overview PDF to: {args.overview_pdf}")


if __name__ == "__main__":
    main()
