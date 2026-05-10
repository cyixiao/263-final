#!/usr/bin/env python3
"""Shared helpers for model scripts."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULT = ROOT / "result"
MODEL_OUT = RESULT
ANALYTIC = DATA / "analytic.csv"

DATA.mkdir(exist_ok=True, parents=True)
RESULT.mkdir(exist_ok=True, parents=True)
MODEL_OUT.mkdir(exist_ok=True, parents=True)
CACHE = ROOT / "archive" / "cache"
CACHE.mkdir(exist_ok=True, parents=True)
os.environ.setdefault("MPLCONFIGDIR", str((CACHE / "mpl").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((CACHE / "xdg").resolve()))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


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


def load_analytic() -> pd.DataFrame:
    if not ANALYTIC.exists():
        raise FileNotFoundError(f"Missing {ANALYTIC}. Run script/prep_data.py first.")
    return pd.read_csv(ANALYTIC)


def get_xy(df: pd.DataFrame, missing_threshold: float = 0.80) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    missing = df[feature_cols].isna().mean()
    feature_cols = [c for c in feature_cols if missing[c] <= missing_threshold]
    return df[feature_cols].copy(), df["log_los"].copy()


def split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype) == "category"]
    numeric = [c for c in X.columns if c not in categorical]
    return numeric, categorical


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric, categorical = split_feature_types(X)
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )


def metrics(y_true: np.ndarray | pd.Series, pred: np.ndarray) -> dict:
    return {
        "RMSE": float(math.sqrt(mean_squared_error(y_true, pred))),
        "MAE": float(mean_absolute_error(y_true, pred)),
        "R2": float(r2_score(y_true, pred)),
    }


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))
