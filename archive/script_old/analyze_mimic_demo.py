#!/usr/bin/env python3
"""Build a demo MIMIC-IV ICU length-of-stay analysis pipeline.

The script intentionally uses only public MIMIC-IV demo files. It mirrors the
project outline so the same pipeline can later be pointed at the full release.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "archive" / "raw_mimic_demo"
OUT = ROOT / "archive" / "old_output" / "legacy_runtime"
LOCAL_PLOTLY = ROOT / ".python_deps_plotly_export"
if LOCAL_PLOTLY.exists():
    sys.path.insert(0, str(LOCAL_PLOTLY))
OUT.mkdir(exist_ok=True, parents=True)
os.environ.setdefault("MPLCONFIGDIR", str((OUT / ".mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((OUT / ".cache").resolve()))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.io as pio
from PIL import Image

from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CHART_ITEMS = {
    "heart_rate": [220045],
    "resp_rate": [220210, 224690, 224689],
    "sbp": [220179, 220050],
    "dbp": [220180, 220051],
    "mbp": [220181, 220052],
    "temperature_c": [223762, 226329],
    "temperature_f": [223761],
    "spo2": [220277],
}

LAB_ITEMS = {
    "hemoglobin": [51222, 50811],
    "creatinine": [50912, 52546],
    "sodium": [50983, 52623, 50824, 52455],
    "potassium": [50971, 52610, 50822, 52452],
    "lactate": [50813, 52442, 53154],
}


def read_csv(rel: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(DATA / rel, **kwargs)


def broad_race(value: object) -> str:
    text = str(value).upper()
    if "WHITE" in text:
        return "WHITE"
    if "BLACK" in text:
        return "BLACK"
    if "HISPANIC" in text or "LATINO" in text:
        return "HISPANIC/LATINO"
    if "ASIAN" in text:
        return "ASIAN"
    if text in {"NAN", "UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"}:
        return "UNKNOWN"
    return "OTHER"


def icd_category(code: object, version: object) -> str | None:
    c = str(code).upper().replace(".", "")
    if not c or c == "NAN":
        return None
    try:
        v = int(version)
    except Exception:
        v = 10 if c[0].isalpha() else 9

    if v == 10:
        if c.startswith("I"):
            return "comorb_cardiovascular"
        if c.startswith("J"):
            return "comorb_respiratory"
        if c.startswith("N"):
            return "comorb_renal"
        if c.startswith("E10") or c.startswith("E11") or c.startswith("E13"):
            return "comorb_diabetes"
        if c.startswith("K7"):
            return "comorb_liver"
        if c.startswith("C"):
            return "comorb_cancer"
        if c.startswith("G") or c.startswith("F"):
            return "comorb_neuro_psych"
        if c.startswith(("A", "B")):
            return "comorb_infectious"
        return None

    if c[0].isdigit():
        n = int(c[:3])
        if 390 <= n <= 459:
            return "comorb_cardiovascular"
        if 460 <= n <= 519:
            return "comorb_respiratory"
        if 580 <= n <= 589:
            return "comorb_renal"
        if c.startswith("250"):
            return "comorb_diabetes"
        if 570 <= n <= 573:
            return "comorb_liver"
        if 140 <= n <= 239:
            return "comorb_cancer"
        if 290 <= n <= 359:
            return "comorb_neuro_psych"
        if 1 <= n <= 139:
            return "comorb_infectious"
    return None


def first_icu_cohort() -> pd.DataFrame:
    patients = read_csv("hosp/patients.csv.gz")
    admissions = read_csv("hosp/admissions.csv.gz", parse_dates=["admittime", "dischtime"])
    icu = read_csv("icu/icustays.csv.gz", parse_dates=["intime", "outtime"])

    icu = icu.sort_values(["subject_id", "intime"]).groupby("subject_id", as_index=False).first()
    cohort = (
        icu.merge(patients, on="subject_id", how="left")
        .merge(admissions, on=["subject_id", "hadm_id"], how="left")
    )
    cohort = cohort[cohort["anchor_age"] >= 18].copy()
    cohort["age"] = cohort["anchor_age"].clip(upper=91)
    cohort["log_los"] = np.log(cohort["los"])
    cohort["race_group"] = cohort["race"].map(broad_race)
    cohort["hospital_expire_flag"] = cohort["hospital_expire_flag"].fillna(0).astype(int)
    return cohort


def add_comorbidities(cohort: pd.DataFrame) -> pd.DataFrame:
    dx = read_csv("hosp/diagnoses_icd.csv.gz")
    dx["category"] = [icd_category(c, v) for c, v in zip(dx["icd_code"], dx["icd_version"])]
    dx = dx.dropna(subset=["category"])
    if dx.empty:
        return cohort

    flags = (
        pd.crosstab([dx["subject_id"], dx["hadm_id"]], dx["category"])
        .clip(upper=1)
        .reset_index()
    )
    out = cohort.merge(flags, on=["subject_id", "hadm_id"], how="left")
    comorb_cols = [c for c in out.columns if c.startswith("comorb_")]
    out[comorb_cols] = out[comorb_cols].fillna(0).astype(int)
    return out


def summarize_events(
    cohort: pd.DataFrame,
    rel: str,
    item_map: Dict[str, list[int]],
    id_cols: list[str],
    value_ranges: Dict[str, tuple[float, float]],
) -> pd.DataFrame:
    wanted = {item: name for name, items in item_map.items() for item in items}
    rows = []
    keep = set(wanted)
    cohort_times = cohort[["subject_id", "hadm_id", "stay_id", "intime"]].copy()

    for chunk in pd.read_csv(DATA / rel, parse_dates=["charttime"], chunksize=200_000):
        chunk = chunk[chunk["itemid"].isin(keep)]
        chunk = chunk.dropna(subset=["valuenum"])
        if chunk.empty:
            continue
        merged = chunk.merge(cohort_times, on=id_cols, how="inner")
        hours = (merged["charttime"] - merged["intime"]).dt.total_seconds() / 3600
        merged = merged[(hours >= 0) & (hours <= 24)].copy()
        if merged.empty:
            continue
        merged["feature"] = merged["itemid"].map(wanted)
        rows.append(merged[["stay_id", "charttime", "feature", "valuenum"]])

    if not rows:
        return pd.DataFrame({"stay_id": cohort["stay_id"]})

    events = pd.concat(rows, ignore_index=True)
    events.loc[events["feature"].eq("temperature_f"), "valuenum"] = (
        events.loc[events["feature"].eq("temperature_f"), "valuenum"] - 32
    ) * 5 / 9
    events.loc[events["feature"].eq("temperature_f"), "feature"] = "temperature_c"

    clean_parts = []
    for feature, group in events.groupby("feature"):
        low, high = value_ranges.get(feature, (-np.inf, np.inf))
        clean_parts.append(group[group["valuenum"].between(low, high)])
    events = pd.concat(clean_parts, ignore_index=True)

    events = events.sort_values(["stay_id", "feature", "charttime"])
    summary = (
        events.groupby(["stay_id", "feature"])["valuenum"]
        .agg(["first", "min", "max", "mean"])
        .reset_index()
    )
    wide = summary.pivot(index="stay_id", columns="feature", values=["first", "min", "max", "mean"])
    wide.columns = [f"{stat}_{feature}" for stat, feature in wide.columns.to_flat_index()]
    return wide.reset_index()


def build_dataset() -> tuple[pd.DataFrame, dict]:
    cohort = add_comorbidities(first_icu_cohort())

    chart_ranges = {
        "heart_rate": (20, 250),
        "resp_rate": (4, 80),
        "sbp": (40, 260),
        "dbp": (20, 160),
        "mbp": (25, 180),
        "temperature_c": (25, 45),
        "spo2": (40, 100),
    }
    lab_ranges = {
        "hemoglobin": (3, 25),
        "creatinine": (0.1, 25),
        "sodium": (90, 180),
        "potassium": (1.5, 9),
        "lactate": (0.1, 30),
    }

    chart_features = summarize_events(
        cohort,
        "icu/chartevents.csv.gz",
        CHART_ITEMS,
        ["subject_id", "hadm_id", "stay_id"],
        chart_ranges,
    )
    lab_features = summarize_events(
        cohort,
        "hosp/labevents.csv.gz",
        LAB_ITEMS,
        ["subject_id", "hadm_id"],
        lab_ranges,
    )

    analytic = cohort.merge(chart_features, on="stay_id", how="left")
    analytic = analytic.merge(lab_features, on="stay_id", how="left", suffixes=("", "_labdup"))
    duplicate_cols = [c for c in analytic.columns if c.endswith("_labdup")]
    analytic = analytic.drop(columns=duplicate_cols)

    meta = {
        "n_demo_subjects": int(read_csv("hosp/patients.csv.gz").shape[0]),
        "n_first_adult_icu_stays": int(analytic.shape[0]),
        "n_features_before_filter": int(analytic.shape[1]),
    }
    return analytic, meta


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "RMSE": float(math.sqrt(mean_squared_error(y_true, pred))),
        "MAE": float(mean_absolute_error(y_true, pred)),
        "R2": float(r2_score(y_true, pred)),
    }


def evaluate_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    id_cols = {
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
    feature_cols = [c for c in df.columns if c not in id_cols]
    missing = df[feature_cols].isna().mean()
    feature_cols = [c for c in feature_cols if missing[c] <= 0.80]
    X = df[feature_cols].copy()
    y = df["log_los"].copy()

    categorical = [c for c in X.columns if X[c].dtype == "object"]
    numeric = [c for c in X.columns if c not in categorical]

    preprocessor = ColumnTransformer(
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=263, shuffle=True
    )
    cv_splits = max(2, min(5, len(X_train) // 4))
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=263)

    models = {
        "Linear regression": LinearRegression(),
        "Elastic net": ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 1.0],
            alphas=np.logspace(-3, 1, 30),
            cv=cv,
            random_state=263,
            max_iter=20000,
        ),
        "Gradient boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=150,
            l2_regularization=0.1,
            min_samples_leaf=5,
            random_state=263,
        ),
        "EBM": ExplainableBoostingRegressor(
            random_state=263,
            interactions=3,
            max_bins=32,
            outer_bags=4,
            n_jobs=1,
        ),
    }

    records = []
    predictions = pd.DataFrame({"y_true": y_test.to_numpy()}, index=y_test.index)
    fitted = {}
    for name, model in models.items():
        if name == "EBM":
            raw_train = X_train.copy()
            raw_test = X_test.copy()
            for col in categorical:
                raw_train[col] = raw_train[col].astype("category")
                raw_test[col] = raw_test[col].astype("category")
            model.fit(raw_train, y_train)
            pred = model.predict(raw_test)
            fitted[name] = model
        else:
            pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            fitted[name] = pipe
        predictions[name] = pred
        row = {"model": name, **metrics(y_test, pred)}
        if name == "Elastic net":
            row["alpha"] = float(pipe.named_steps["model"].alpha_)
            row["l1_ratio"] = float(pipe.named_steps["model"].l1_ratio_)
        records.append(row)

    performance = pd.DataFrame(records).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    best_name = performance.loc[0, "model"]
    best = fitted[best_name]
    X_perm = X_test.copy()
    if best_name == "EBM":
        for col in categorical:
            X_perm[col] = X_perm[col].astype("category")

    perm = permutation_importance(
        best,
        X_perm if best_name == "EBM" else X_test,
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

    context = {
        "feature_cols": feature_cols,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "best_model": str(best_name),
        "cv_splits": int(cv_splits),
    }
    return performance, predictions, importance, context, fitted


def table_one(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in [
        ("age", "Age, years"),
        ("los", "ICU length of stay, days"),
        ("log_los", "Log ICU length of stay"),
    ]:
        rows.append(
            {
                "Variable": label,
                "Summary": f"{df[col].mean():.2f} ({df[col].std():.2f})",
            }
        )
    for col, label in [
        ("gender", "Female sex"),
        ("hospital_expire_flag", "In-hospital mortality"),
    ]:
        if col == "gender":
            value = (df[col] == "F").mean()
        else:
            value = df[col].mean()
        rows.append({"Variable": label, "Summary": f"{value * 100:.1f}%"})
    for col in [c for c in df.columns if c.startswith("comorb_")]:
        rows.append({"Variable": col, "Summary": f"{df[col].mean() * 100:.1f}%"})
    return pd.DataFrame(rows)


def write_plots(performance: pd.DataFrame, predictions: pd.DataFrame, importance: pd.DataFrame, best: str) -> None:
    OUT.mkdir(exist_ok=True, parents=True)

    plt.figure(figsize=(7, 4.5))
    plt.barh(performance["model"], performance["RMSE"], color="#356b8c")
    plt.gca().invert_yaxis()
    plt.xlabel("Test RMSE on log ICU LOS")
    plt.title("Model comparison on MIMIC-IV demo")
    plt.tight_layout()
    plt.savefig(OUT / "model_performance.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(predictions["y_true"], predictions[best], s=38, alpha=0.8, color="#2f6f4e")
    low = min(predictions["y_true"].min(), predictions[best].min())
    high = max(predictions["y_true"].max(), predictions[best].max())
    plt.plot([low, high], [low, high], color="#444444", linewidth=1)
    plt.xlabel("Observed log ICU LOS")
    plt.ylabel("Predicted log ICU LOS")
    plt.title(f"Observed vs predicted: {best}")
    plt.tight_layout()
    plt.savefig(OUT / "observed_vs_predicted.png", dpi=200)
    plt.close()

    top = importance.head(12).iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_sd"], color="#8c5a35")
    plt.xlabel("Permutation importance: RMSE improvement")
    plt.title(f"Top features for {best}")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=200)
    plt.close()


def safe_name(value: str, limit: int = 90) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))[:limit]


def chrome_screenshot_plotly(fig, output_png: Path, width: int = 1100, height: int = 760) -> None:
    chrome_candidates = [
        ROOT / ".chrome_for_kaleido/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    ]
    chrome = next((path for path in chrome_candidates if path.exists()), None)
    if chrome is None:
        raise RuntimeError("Chrome is required to screenshot Plotly figures when Kaleido is unavailable.")

    html_path = output_png.with_suffix(".html")
    fig.write_html(str(html_path), include_plotlyjs=True, full_html=True)
    cmd = [
        str(chrome),
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--hide-scrollbars",
        f"--window-size={width},{height}",
        "--virtual-time-budget=4000",
        f"--screenshot={output_png}",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    html_path.unlink(missing_ok=True)


def plotly_to_png(fig, output_png: Path, width: int = 1100, height: int = 760) -> None:
    if getattr(plotly_to_png, "_kaleido_failed", False):
        chrome_screenshot_plotly(fig, output_png, width=width, height=height)
        return
    try:
        pio.write_image(fig, str(output_png), width=width, height=height, scale=2)
    except Exception:
        plotly_to_png._kaleido_failed = True
        chrome_screenshot_plotly(fig, output_png, width=width, height=height)


def write_ebm_visualizations(ebm: ExplainableBoostingRegressor | None) -> list[str]:
    if ebm is None:
        return []
    pdf_dir = OUT / "ebm_global_explanations_pdf"
    temp_folder = OUT / "temp_ebm_plots"
    pdf_dir.mkdir(exist_ok=True, parents=True)
    temp_folder.mkdir(exist_ok=True, parents=True)
    explanation = ebm.explain_global(name="EBM ICU LOS")
    written = []
    plot_specs: list[tuple[int | None, str, Path, Path]] = [
        (None, "overview", pdf_dir / "00_overview.pdf", temp_folder / "00_overview.png")
    ]

    term_names = list(getattr(ebm, "term_names_", []))
    for idx, term_name in enumerate(term_names):
        i = idx + 1
        stem = f"{i:02d}_{safe_name(term_name)}"
        plot_specs.append((idx, str(term_name), pdf_dir / f"{stem}.pdf", temp_folder / f"{stem}.png"))

    try:
        for index, title, pdf_path, png_path in plot_specs:
            fig_plotly = explanation.visualize() if index is None else explanation.visualize(index)
            plotly_to_png(fig_plotly, png_path)

            img = Image.open(png_path)
            fig, ax = plt.subplots(figsize=(11, 7.6))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(title, fontsize=12)
            plt.tight_layout(pad=0.3)
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
            plt.close(fig)
            written.append(str(pdf_path))

        n_features = len(plot_specs)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
        axes = np.asarray(axes).flatten()

        for i, (_, title, _, png_path) in enumerate(plot_specs):
            img = Image.open(png_path)
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(title, fontsize=9)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.tight_layout(pad=0.3)
        combined_pdf = OUT / "ebm_global_explanations_combined_3_per_row.pdf"
        fig.savefig(combined_pdf, format="pdf", bbox_inches="tight")
        plt.close(fig)
        written.append(str(combined_pdf))
    finally:
        shutil.rmtree(temp_folder, ignore_errors=True)
    return written


def markdown_table(df: pd.DataFrame, floatfmt: str | None = None) -> str:
    display = df.copy()
    if floatfmt is not None:
        for col in display.columns:
            if pd.api.types.is_numeric_dtype(display[col]):
                display[col] = display[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
    display = display.fillna("")
    headers = [str(c) for c in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in display.columns) + " |")
    return "\n".join(lines)


def write_report(
    analytic: pd.DataFrame,
    table1: pd.DataFrame,
    performance: pd.DataFrame,
    importance: pd.DataFrame,
    meta: dict,
    context: dict,
    ebm_visual_files: list[str],
) -> None:
    OUT.mkdir(exist_ok=True, parents=True)
    perf_md = markdown_table(performance, ".3f")
    table1_md = markdown_table(table1)
    top_md = markdown_table(importance.head(10), ".4f")
    feature_list = ", ".join(context["feature_cols"])
    ebm_visual_note = "\n".join(
        f"- `{Path(path).relative_to(OUT)}`" for path in ebm_visual_files[:20]
    )
    if len(ebm_visual_files) > 20:
        ebm_visual_note += f"\n- ... plus {len(ebm_visual_files) - 20} more files"

    report = f"""# Predicting ICU Length of Stay Using the MIMIC-IV Clinical Database Demo

## Executive Summary

This analysis implements the project outline on the public MIMIC-IV Clinical Database Demo v2.2. The demo was used to build and test an end-to-end pipeline: cohort construction, first-24-hour feature extraction, preprocessing, model training, model comparison, and interpretation. Because the demo contains only a small subset of patients, all performance estimates below should be treated as a pipeline validation exercise rather than final scientific evidence.

## Data and Cohort

- Source: MIMIC-IV Clinical Database Demo v2.2.
- Raw demo subjects available in `patients`: {meta["n_demo_subjects"]}.
- Analytic cohort: adult patients' first ICU stay only.
- Final analytic sample: {meta["n_first_adult_icu_stays"]} ICU stays.
- Train/test split: {context["n_train"]}/{context["n_test"]} stays.
- Outcome: log-transformed ICU length of stay in days, using `icustays.los`.

## Table 1

{table1_md}

## Feature Construction

Predictors follow the project outline: demographics, insurance/race/admission context, diagnosis-based comorbidity indicators, and first-24-hour vital/lab summaries. Repeated measurements were summarized with first, minimum, maximum, and mean values. Implausible physiologic values were removed using broad clinical ranges. Features with more than 80% missingness in the demo were excluded before modeling.

Features retained for modeling:

`{feature_list}`

## Models

Four approaches were fit on the same preprocessing pipeline:

1. Simple linear regression baseline.
2. Elastic net regression with cross-validated penalty strength and mixing parameter.
3. Gradient boosting using scikit-learn's histogram gradient boosting regressor.
4. Explainable boosting machine (EBM) using the `interpret` package.

Categorical variables were one-hot encoded. Numeric variables were median-imputed with missingness indicators and standardized. Tuning for elastic net used {context["cv_splits"]}-fold cross-validation on the training set. Final performance was evaluated on the held-out test set.

## Model Performance

{perf_md}

The lowest test RMSE model in this demo run was **{context["best_model"]}**. With the small demo sample, differences between models should not be over-interpreted. The main value here is that the same code path can be re-run once the full MIMIC-IV data are available.

## Feature Importance

Permutation importance for the selected model:

{top_md}

![Model performance](model_performance.png)

![Observed vs predicted](observed_vs_predicted.png)

![Feature importance](feature_importance.png)

## EBM Global Explanation Visualizations

The EBM global explanation was exported as clean matplotlib-based PDF files under `outputs/ebm_global_explanations_pdf/`. The overview page gives term-level importance, and the feature-specific pages show the learned contribution function for each EBM term. A combined PDF with three plots per row was also exported as `outputs/ebm_global_explanations_combined_3_per_row.pdf`.

{ebm_visual_note}

Note: the `interpret` package warns that its EBM plots do not display missing-value bins directly. The EBM still models missingness internally; if we want the missing-value contribution shown visually in the final full-data report, we should either encode missing values explicitly before fitting EBM or inspect `term_scores_` for the missing bin.

## Interpretation

The demo confirms that the proposed analytic structure is feasible: first ICU stays can be linked to demographics, admissions, diagnoses, charted vital signs, and laboratory results; first-24-hour summaries can be created; and regression, penalized regression, flexible tree-based learning, and EBM models can all be evaluated under a common framework.

The result should not be used to make clinical claims. The public demo is deliberately small, so estimates are unstable, test-set size is limited, and rare comorbidities/labs may be absent or heavily missing. On the full MIMIC-IV release, we should revisit missingness thresholds, tune gradient boosting and EBM more carefully, add richer comorbidity definitions, and perform subgroup/fairness checks by sex, race group, insurance, and age category.

## Reproducibility

Run:

```bash
python3 scripts/analyze_mimic_demo.py
```

Expected outputs:

- `outputs/analytic_dataset.csv`
- `outputs/table1.csv`
- `outputs/model_performance.csv`
- `outputs/feature_importance.csv`
- `outputs/model_performance.png`
- `outputs/observed_vs_predicted.png`
- `outputs/feature_importance.png`
- `outputs/mimic_demo_report.md`
"""
    (OUT / "mimic_demo_report.md").write_text(report)


def main() -> None:
    OUT.mkdir(exist_ok=True, parents=True)
    analytic, meta = build_dataset()
    table1 = table_one(analytic)
    performance, predictions, importance, context, fitted = evaluate_models(analytic)
    write_plots(performance, predictions, importance, context["best_model"])
    ebm_visual_files = write_ebm_visualizations(fitted.get("EBM"))

    analytic.to_csv(OUT / "analytic_dataset.csv", index=False)
    table1.to_csv(OUT / "table1.csv", index=False)
    performance.to_csv(OUT / "model_performance.csv", index=False)
    predictions.to_csv(OUT / "test_predictions.csv", index=True)
    importance.to_csv(OUT / "feature_importance.csv", index=False)
    (OUT / "run_metadata.json").write_text(json.dumps({**meta, **context}, indent=2))
    write_report(analytic, table1, performance, importance, meta, context, ebm_visual_files)

    print(json.dumps({**meta, **context, "outputs": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
