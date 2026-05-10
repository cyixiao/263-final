#!/usr/bin/env python3
"""Run gradient boosting model for ICU LOS prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import MODEL_OUT, get_xy, load_analytic, make_preprocessor, metrics, save_json


def main() -> None:
    df = load_analytic()
    X, y = get_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=263, shuffle=True)

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=150,
        l2_regularization=0.1,
        min_samples_leaf=5,
        random_state=263,
    )
    pipe = Pipeline([("preprocess", make_preprocessor(X)), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    out_dir = MODEL_OUT / "gbm"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": "gbm", **metrics(y_test, pred)}]).to_csv(out_dir / "perf.csv", index=False)
    pd.DataFrame({"y_true": y_test.to_numpy(), "gbm": pred}, index=y_test.index).to_csv(
        out_dir / "pred.csv", index=True
    )
    save_json(out_dir / "meta.json", {"n_train": len(X_train), "n_test": len(X_test)})
    print(f"Saved gradient boosting results to {out_dir}")


if __name__ == "__main__":
    main()
