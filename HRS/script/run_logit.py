#!/usr/bin/env python3
"""Run logistic regression baselines for HRS diabetes prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import RESULT, classification_metrics, get_xy, load_analytic, make_preprocessor, save_json


def main() -> None:
    df = load_analytic()
    X, y = get_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=263, stratify=y)

    out_dir = RESULT / "logit"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "logit": LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs"),
        "elastic_net_logit": LogisticRegressionCV(
            Cs=10,
            cv=5,
            penalty="elasticnet",
            solver="saga",
            l1_ratios=[0.1, 0.5, 0.9],
            scoring="roc_auc",
            max_iter=5000,
            class_weight="balanced",
            random_state=263,
            n_jobs=-1,
        ),
    }

    perf_rows = []
    pred_df = pd.DataFrame({"y_true": y_test.to_numpy()}, index=y_test.index)
    meta = {"n_train": len(X_train), "n_test": len(X_test)}
    for name, model in models.items():
        pipe = Pipeline([("preprocess", make_preprocessor(X)), ("model", model)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred_df[name] = proba
        perf_rows.append({"model": name, **classification_metrics(y_test, proba)})
        if name == "elastic_net_logit":
            meta["elastic_net_C"] = float(pipe.named_steps["model"].C_[0])
            meta["elastic_net_l1_ratio"] = float(pipe.named_steps["model"].l1_ratio_[0])

    pd.DataFrame(perf_rows).to_csv(out_dir / "perf.csv", index=False)
    pred_df.to_csv(out_dir / "pred.csv", index=True)
    save_json(out_dir / "meta.json", meta)
    print(f"Saved logistic regression results to {out_dir}")


if __name__ == "__main__":
    main()
