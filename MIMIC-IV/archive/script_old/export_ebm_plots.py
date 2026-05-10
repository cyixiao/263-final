#!/usr/bin/env python3
"""Export official InterpretML EBM global explanation plots to PDF.

This script intentionally follows the same pattern as the user's prior notebook:

1. Fit an ExplainableBoostingRegressor.
2. Call ``ebm_global.visualize(i)`` for each EBM term.
3. Save each Plotly figure to a temporary PNG with ``plotly.io.write_image``.
4. Place the PNGs into matplotlib figures.
5. Save one PDF per term and one combined PDF with three plots per row.

It does not use Chrome headless fallbacks. If ``pio.write_image`` fails, fix the
Plotly/Kaleido environment and rerun this script.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
from PIL import Image
from interpret.glassbox import ExplainableBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
ANALYTIC = OUT / "analytic_dataset.csv"


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


def build_ebm_input(missing_threshold: float = 0.80) -> tuple[pd.DataFrame, pd.Series]:
    if not ANALYTIC.exists():
        raise FileNotFoundError(
            f"Missing {ANALYTIC}. Run scripts/analyze_mimic_demo.py once first."
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


def save_single_pdf_from_png(png_path: Path, pdf_path: Path, title: str) -> None:
    img = Image.open(png_path)
    fig, ax = plt.subplots(figsize=(11, 7.6))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    plt.tight_layout(pad=0.3)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def export_ebm_pdfs(
    ebm: ExplainableBoostingRegressor,
    separate_dir: Path,
    combined_pdf: Path,
    temp_dir: Path,
    n_cols: int = 3,
) -> None:
    separate_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    ebm_global = ebm.explain_global()
    specs: list[tuple[str, object, Path, Path]] = []

    overview_png = temp_dir / "00_overview.png"
    overview_pdf = separate_dir / "00_overview.pdf"
    specs.append(("overview", ebm_global.visualize(), overview_png, overview_pdf))

    for i, feature_name in enumerate(ebm_global.feature_names):
        stem = f"{i + 1:02d}_{safe_name(feature_name)}"
        png_path = temp_dir / f"{stem}.png"
        pdf_path = separate_dir / f"{stem}.pdf"
        specs.append((str(feature_name), ebm_global.visualize(i), png_path, pdf_path))

    try:
        for title, fig_plotly, png_path, pdf_path in specs:
            pio.write_image(fig_plotly, str(png_path))
            save_single_pdf_from_png(png_path, pdf_path, title)

        n_features = len(specs)
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
        axes = np.asarray(axes).flatten()

        for i, (title, _, png_path, _) in enumerate(specs):
            img = Image.open(png_path)
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(title, fontsize=9)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.tight_layout(pad=0.3)
        fig.savefig(combined_pdf, format="pdf", bbox_inches="tight")
        plt.close(fig)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--separate-dir",
        type=Path,
        default=OUT / "ebm_official_style_pdfs",
        help="Directory for one PDF per EBM term.",
    )
    parser.add_argument(
        "--combined-pdf",
        type=Path,
        default=OUT / "ebm_official_style_combined_3_per_row.pdf",
        help="Combined PDF path.",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=OUT / "temp_ebm_official_plots",
        help="Temporary PNG directory; deleted after export.",
    )
    args = parser.parse_args()

    X, y = build_ebm_input()
    ebm = fit_ebm(X, y)
    export_ebm_pdfs(ebm, args.separate_dir, args.combined_pdf, args.temp_dir)

    print(f"Saved separate PDFs to: {args.separate_dir}")
    print(f"Saved combined PDF to: {args.combined_pdf}")


if __name__ == "__main__":
    main()
