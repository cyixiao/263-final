# HRS Diabetes Prediction

This folder contains the HRS version of our project. The task is to analyze diabetes risk using demographic, socioeconomic, BMI, smoking, urbanicity, and physical activity variables from HRS Wave 10. The focus is interpretation and public-health insight, not just prediction score.

The outcome is:

- `diabetes`: 0 = no diabetes, 1 = diabetes

The raw HRS analytic file is kept in:

```text
archive/HRS_IPW.dta
```

The processed file used by the model scripts is:

```text
data/analytic.csv
```

## Structure

```text
script/      analysis scripts
data/        processed data and summary tables
result/      model outputs, predictions, and figures
archive/     original HRS analytic .dta file
report/      LaTeX report project and final PDF
```

## Scripts

Run from the `HRS/` folder.

On this machine, use the Anaconda Python environment:

```bash
/Users/cyixiao/anaconda3/bin/python3 script/prep_data.py
```

```bash
/Users/cyixiao/anaconda3/bin/python3 script/prep_data.py
/Users/cyixiao/anaconda3/bin/python3 script/run_logit.py
/Users/cyixiao/anaconda3/bin/python3 script/run_rf.py
/Users/cyixiao/anaconda3/bin/python3 script/run_gbm.py
/Users/cyixiao/anaconda3/bin/python3 script/run_ebm.py
/Users/cyixiao/anaconda3/bin/python3 script/run_nn.py
/Users/cyixiao/anaconda3/bin/python3 script/run_spline.py
/Users/cyixiao/anaconda3/bin/python3 script/run_subgroup.py
/Users/cyixiao/anaconda3/bin/python3 script/run_ipw.py
/Users/cyixiao/anaconda3/bin/python3 script/calibrate.py
/Users/cyixiao/anaconda3/bin/python3 script/combine.py
```

Main scripts:

- `prep_data.py`: creates `data/analytic.csv`
- `run_logit.py`: logistic regression and elastic net logistic regression
- `run_rf.py`: random forest classifier
- `run_gbm.py`: gradient boosting classifier
- `run_ebm.py`: EBM classifier and EBM plots
- `run_nn.py`: PyTorch MLP classifier
- `run_spline.py`: restricted cubic spline logistic model for age and BMI curves
- `run_subgroup.py`: subgroup and interaction analysis for physical activity
- `run_ipw.py`: IPW-weighted logistic and EBM analyses
- `calibrate.py`: calibration, ROC, and PR curves
- `combine.py`: combined model performance summary

## Results

Model outputs are saved by method:

```text
result/logit/
result/rf/
result/gbm/
result/ebm/
result/nn/
result/spline/
result/subgroup/
result/ipw/
result/calib/
result/summary/
```

The main comparison table is:

```text
result/summary/perf.csv
```

Main metrics:

- AUC
- PR-AUC
- Accuracy
- Sensitivity
- Specificity
- F1
- Brier score
- Log loss

For prediction models, `pr_treat` and `ipw` are kept in the data file but excluded from the default feature set because they are causal/IPW analysis variables rather than ordinary baseline predictors.

## Report

The final LaTeX report is in:

```text
report/report.tex
report/report.pdf
```

The report figures and tables are stored in `report/figures/` and `report/tables/`.
