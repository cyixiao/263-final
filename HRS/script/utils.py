#!/usr/bin/env python3
"""Shared helpers for HRS classification scripts."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULT = ROOT / "result"
ARCHIVE = ROOT / "archive"
RAW = ARCHIVE / "HRS_IPW.dta"
ANALYTIC = DATA / "analytic.csv"

DATA.mkdir(exist_ok=True, parents=True)
RESULT.mkdir(exist_ok=True, parents=True)
ARCHIVE.mkdir(exist_ok=True, parents=True)
CACHE = ARCHIVE / "cache"
CACHE.mkdir(exist_ok=True, parents=True)
os.environ.setdefault("MPLCONFIGDIR", str((CACHE / "mpl").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((CACHE / "xdg").resolve()))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


ID_COLS = {"hhidpn"}
OUTCOME = "diabetes"
EXCLUDE_FROM_PREDICTION = {"pr_treat", "ipw"}
CATEGORICAL = [
    "physical_act",
    "female",
    "raceeth",
    "educ",
    "marital_status",
    "income",
    "smoking",
    "urban",
]
NUMERIC = ["age", "bmi"]


def load_analytic() -> pd.DataFrame:
    if not ANALYTIC.exists():
        raise FileNotFoundError(f"Missing {ANALYTIC}. Run script/prep_data.py first.")
    return pd.read_csv(ANALYTIC)


def get_xy(df: pd.DataFrame, include_weights: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = ID_COLS | {OUTCOME}
    if not include_weights:
        drop_cols |= EXCLUDE_FROM_PREDICTION
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return df[feature_cols].copy(), df[OUTCOME].astype(int).copy()


def split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [c for c in CATEGORICAL if c in X.columns]
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
                        ("imputer", SimpleImputer(strategy="median")),
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


def classification_metrics(y_true: np.ndarray | pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_true_arr = np.asarray(y_true).astype(int)
    proba_arr = np.clip(np.asarray(proba), 1e-6, 1 - 1e-6)
    pred = (proba_arr >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else math.nan
    return {
        "AUC": float(roc_auc_score(y_true_arr, proba_arr)),
        "PR_AUC": float(average_precision_score(y_true_arr, proba_arr)),
        "Accuracy": float(accuracy_score(y_true_arr, pred)),
        "Sensitivity": float(recall_score(y_true_arr, pred, zero_division=0)),
        "Specificity": float(specificity),
        "Precision": float(precision_score(y_true_arr, pred, zero_division=0)),
        "F1": float(f1_score(y_true_arr, pred, zero_division=0)),
        "Brier": float(brier_score_loss(y_true_arr, proba_arr)),
        "LogLoss": float(log_loss(y_true_arr, proba_arr)),
        "threshold": float(threshold),
    }


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))
