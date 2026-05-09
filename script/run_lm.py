#!/usr/bin/env python3
"""Run linear regression and elastic net models for ICU LOS prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import MODEL_OUT, get_xy, load_analytic, make_preprocessor, metrics, save_json


def main() -> None:
    df = load_analytic()
    X, y = get_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=263, shuffle=True)
    cv_splits = max(2, min(5, len(X_train) // 4))
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=263)

    models = {
        "linear_regression": LinearRegression(),
        "elastic_net": ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 1.0],
            alphas=np.logspace(-3, 1, 30),
            cv=cv,
            random_state=263,
            max_iter=20000,
        ),
    }

    out_dir = MODEL_OUT / "lm"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    predictions = pd.DataFrame({"y_true": y_test.to_numpy()}, index=y_test.index)

    for name, model in models.items():
        pipe = Pipeline([("preprocess", make_preprocessor(X)), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        predictions[name] = pred
        row = {"model": name, **metrics(y_test, pred)}
        if name == "elastic_net":
            row["alpha"] = float(pipe.named_steps["model"].alpha_)
            row["l1_ratio"] = float(pipe.named_steps["model"].l1_ratio_)
        records.append(row)

    pd.DataFrame(records).to_csv(out_dir / "perf.csv", index=False)
    predictions.to_csv(out_dir / "pred.csv", index=True)
    save_json(out_dir / "meta.json", {"n_train": len(X_train), "n_test": len(X_test), "cv_splits": cv_splits})
    print(f"Saved linear/elastic net results to {out_dir}")


if __name__ == "__main__":
    main()
