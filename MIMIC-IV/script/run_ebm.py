#!/usr/bin/env python3
"""Run EBM model and export EBM explanation figures."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ebm_plot import export_manual_plots
from utils import MODEL_OUT, get_xy, load_analytic, metrics, save_json, split_feature_types


def main() -> None:
    df = load_analytic()
    X, y = get_xy(df)
    _, categorical = split_feature_types(X)
    for col in categorical:
        X[col] = X[col].astype("category")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=263, shuffle=True)
    model = ExplainableBoostingRegressor(
        random_state=263,
        interactions=3,
        max_bins=32,
        outer_bags=4,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    out_dir = MODEL_OUT / "ebm"
    fig_dir = out_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{"model": "ebm", **metrics(y_test, pred)}]).to_csv(out_dir / "perf.csv", index=False)
    pd.DataFrame({"y_true": y_test.to_numpy(), "ebm": pred}, index=y_test.index).to_csv(
        out_dir / "pred.csv", index=True
    )

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=30,
        random_state=263,
        scoring="neg_root_mean_squared_error",
    )
    importance = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": perm.importances_mean,
                "importance_sd": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(out_dir / "importance.csv", index=False)

    export_manual_plots(
        model,
        separate_dir=fig_dir,
        combined_pdf=out_dir / "fig_all.pdf",
        overview_pdf=out_dir / "fig_global.pdf",
    )
    save_json(out_dir / "meta.json", {"n_train": len(X_train), "n_test": len(X_test)})
    print(f"Saved EBM results and figures to {out_dir}")


if __name__ == "__main__":
    main()
