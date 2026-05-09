# ICU Length of Stay Prediction

This repo contains our demo pipeline for predicting ICU length of stay using the public MIMIC-IV Clinical Database Demo v2.2.

The current version is mainly a proof of concept. The demo dataset only has 100 patients, so the results should not be treated as final scientific conclusions. The goal is to make sure the full workflow runs before we move to the full MIMIC-IV data.

## Project Goal

We use information available during the first 24 hours after ICU admission to predict ICU length of stay.

The outcome is:

- `log(ICU length of stay)`

The predictors include:

- demographics: age, sex, race
- admission information: admission type, insurance, admission location
- ICD-based comorbidity indicators
- first-24-hour vital signs and labs, summarized by first, min, max, and mean

## Folder Structure

```text
script/      analysis scripts
data/        processed analytic data
result/      model outputs and figures
archive/     raw demo data and old scripts
```

Important files:

```text
data/analytic.csv          final analysis dataset
data/table1.csv            summary table
result/summary/perf.csv    model comparison table
result/summary/perf.png    model comparison plot
```

## Scripts

Run the scripts from the repo root.

```bash
python3 script/prep_data.py
python3 script/run_lm.py
python3 script/run_gbm.py
python3 script/run_rf.py
python3 script/run_ebm.py
python3 script/combine.py
```

What each script does:

- `script/prep_data.py`: builds the analysis dataset from the MIMIC-IV demo files
- `script/run_lm.py`: runs linear regression and elastic net
- `script/run_gbm.py`: runs gradient boosting
- `script/run_rf.py`: runs random forest
- `script/run_ebm.py`: runs EBM and saves EBM explanation figures
- `script/combine.py`: combines model performance into one summary table
- `script/ebm_plot.py`: helper functions for EBM plots
- `script/utils.py`: shared helper functions

## Models

We currently compare four methods:

- linear regression
- elastic net regression
- gradient boosting
- random forest
- explainable boosting machine, or EBM

The main metrics are:

- RMSE
- MAE
- R2

All metrics are calculated on the held-out test set.

## Current Demo Results

Current model comparison is saved in:

```text
result/summary/perf.csv
```

At this stage, elastic net performs best on the demo data by RMSE. EBM is close and is useful because it gives interpretable feature effect plots.

## EBM Results

EBM outputs are in:

```text
result/ebm/
```

Useful files:

```text
result/ebm/perf.csv          EBM performance
result/ebm/importance.csv    permutation importance
result/ebm/fig_global.pdf    global term importance
result/ebm/fig_all.pdf       combined editable EBM plots
result/ebm/figs/             individual editable EBM plots
result/ebm/official/         official-style EBM plots
```

## Notes

The raw public demo files are kept in:

```text
archive/raw_mimic_demo/
```

Old scripts are kept in:

```text
archive/script_old/
```

When we get access to the full MIMIC-IV data, the main changes should be in the data preparation step. The model scripts can mostly stay the same.
