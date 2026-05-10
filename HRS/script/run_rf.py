#!/usr/bin/env python3
"""Run random forest classifier for HRS diabetes prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import RESULT, classification_metrics, get_xy, load_analytic, make_preprocessor, save_json


def main() -> None:
    df = load_analytic()
    X, y = get_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=263, stratify=y)

    model = RandomForestClassifier(
        n_estimators=600,
        max_features="sqrt",
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=263,
        n_jobs=-1,
    )
    pipe = Pipeline([("preprocess", make_preprocessor(X)), ("model", model)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]

    out_dir = RESULT / "rf"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": "rf", **classification_metrics(y_test, proba)}]).to_csv(out_dir / "perf.csv", index=False)
    pd.DataFrame({"y_true": y_test.to_numpy(), "rf": proba}, index=y_test.index).to_csv(out_dir / "pred.csv", index=True)

    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    importance = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    importance.to_csv(out_dir / "importance.csv", index=False)
    save_json(out_dir / "meta.json", {"n_train": len(X_train), "n_test": len(X_test), "n_estimators": 600})
    print(f"Saved random forest results to {out_dir}")


if __name__ == "__main__":
    main()
