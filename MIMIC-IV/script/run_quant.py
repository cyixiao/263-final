#!/usr/bin/env python3
"""Run quantile regression models for ICU LOS prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import MODEL_OUT, get_xy, load_analytic, make_preprocessor, metrics, save_json


QUANTILES = [0.10, 0.50, 0.75, 0.90]


def main() -> None:
    df = load_analytic()
    X, y = get_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=263, shuffle=True)

    out_dir = MODEL_OUT / "quant"
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame({"y_true": y_test.to_numpy()}, index=y_test.index)
    records = []
    for q in QUANTILES:
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=q,
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=5,
            random_state=263,
        )
        pipe = Pipeline([("preprocess", make_preprocessor(X)), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        col = f"q{int(q * 100):02d}"
        predictions[col] = pred
        records.append(
            {
                "model": "quant",
                "quantile": q,
                "pinball_loss": float(mean_pinball_loss(y_test, pred, alpha=q)),
            }
        )

    median_pred = predictions["q50"].to_numpy()
    perf = {"model": "quant_median", **metrics(y_test, median_pred)}
    perf["coverage_q10_q90"] = float(((y_test >= predictions["q10"]) & (y_test <= predictions["q90"])).mean())
    perf["mean_interval_width_q10_q90"] = float((predictions["q90"] - predictions["q10"]).mean())

    pd.DataFrame([perf]).to_csv(out_dir / "perf.csv", index=False)
    pd.DataFrame(records).to_csv(out_dir / "pinball.csv", index=False)
    predictions.to_csv(out_dir / "pred.csv", index=True)
    save_json(
        out_dir / "meta.json",
        {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "quantiles": QUANTILES,
            "outcome_scale": "log ICU LOS",
        },
    )
    print(f"Saved quantile regression results to {out_dir}")


if __name__ == "__main__":
    main()
