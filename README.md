# Credit Scoring Pipeline

Production-style credit scoring project for digital lending: application-time
default risk, a gradient-boosted champion, validation, monitoring, and a
FastAPI scoring service.

## Problem framing

A telecom and mobile-money lender wants to automate approve, refer, and
decline decisions for digital loans. The model ranks default risk at
application time using repayment history, transactional behaviour, and
bureau-style attributes.

This repository uses the public **Home Credit Default Risk** dataset from Kaggle
as a stand-in for proprietary lending data. Home Credit is consumer credit data,
not telecom or mobile-money transaction data.

## Status

Stages 1-9 are implemented and runnable from the Makefile: data download,
data quality, DuckDB features, stratified splits, model comparison, champion
training, evaluation, explainability, drift monitoring, and the scoring API.
`reports/metrics.json` (from `make evaluate`) is the source of truth for the
champion numbers below. Notebook walkthroughs live in `docs/`.

## Architecture

```
data/raw CSVs
    |
    v
src/data_quality          clean application and child tables
    |
    v
sql/*.sql + src/features  one row per SK_ID_CURR
    |
    v
src/preprocess            stratified 80/20 holdout, 5-fold CV,
                          train-only impute / cap / missing flags
    |
    v
src/train                 champion pipeline -> models/
    |
    +--> src/evaluate     reports/metrics.json, model card, figures
    +--> src/explain      SHAP report and figures
    +--> src/monitor      score PSI and feature CSI
    |
    v
src/scoring + api/main    /health and /score
```

Preprocessing is nested inside each model pipeline and refit on training folds
only. Child-table SQL keeps relative day and month offsets at or before
application time (`tests/test_feature_sql_leakage.py`).

## Results

Champion: LightGBM (`scale_pos_weight` for an ~8% default rate), version
0.1.0. Train 246005 rows, holdout 61502 rows, 52 features. Holdout numbers
from `reports/metrics.json`:

| Metric | Holdout |
| --- | --- |
| ROC AUC | 0.772547 |
| Gini | 0.545093 |
| KS | 0.417136 |
| PR AUC | 0.266848 |
| Log loss | 0.542189 |
| Brier | 0.183511 |

PR AUC is the ranking metric that respects the low default rate. The output is
an uncalibrated risk score, not a probability: class weighting moves the
scores. Do not plug them into a pricing formula.

The same holdout, from `reports/model_comparison.json`, ranked the family
winners as LightGBM 0.772547, XGBoost 0.768142, CatBoost 0.767492, random
forest 0.764749, decision tree 0.716230, logistic regression 0.712802.
Random forest had the lowest Brier (0.116261) and is the better-calibrated
baseline. LightGBM was selected for discrimination.

### Cutoff table (illustrative)

Approve when the risk score is below the threshold. The loss proxy is
`approval_rate * bad_rate_approved * LGD` with exposure 1 and LGD 1. It
ignores interest income. At the reported max-F1 threshold of 0.65 the holdout
approves 84.6% of applicants with a 5.0% bad rate among approved.

| Threshold | Approval rate | Bad rate approved | Loss proxy |
| --- | --- | --- | --- |
| 0.20 | 0.243390 | 0.016902 | 0.004114 |
| 0.35 | 0.485594 | 0.026017 | 0.012634 |
| 0.50 | 0.684742 | 0.035690 | 0.024438 |
| 0.65 | 0.845517 | 0.050230 | 0.042470 |
| 0.80 | 0.964001 | 0.069002 | 0.066518 |

The 0.65 threshold maximises F1 on the holdout, so that operating point is
optimistic. ROC AUC, Gini, KS, and PR AUC do not depend on it.

### Segment stability

Holdout ROC AUC stays in a narrow band. Younger applicants default more often.
Income quartiles use the training distribution; income is lumpy, so the
quartile counts are unequal.

| Age | n | Default rate | ROC AUC |
| --- | --- | --- | --- |
| under 30 | 9180 | 0.111111 | 0.746488 |
| 30 to 40 | 16542 | 0.096784 | 0.776359 |
| 40 to 50 | 15232 | 0.076681 | 0.776897 |
| 50 plus | 20548 | 0.057232 | 0.747065 |

| Income quartile | n | Default rate | ROC AUC |
| --- | --- | --- | --- |
| q1 | 20198 | 0.084959 | 0.772208 |
| q2 | 11021 | 0.084021 | 0.770258 |
| q3 | 15980 | 0.083104 | 0.773435 |
| q4 | 14303 | 0.069566 | 0.770832 |

## Key findings

- External bureau scores (`app_ext_source_2`, then 3 and 1) dominate the SHAP
  explanation. A stronger external score lowers predicted risk.
- Boosting beats a single tree and a linear model by a clear margin. The
  forest is close on AUC and better calibrated.
- Train-versus-holdout score PSI on the champion is near zero (stable). A
  deliberately shifted holdout in notebook 12 pushes score PSI into alert and
  flags the features that were moved.
- Native LightGBM tree SHAP reconstructs the model output to numerical noise,
  and the same values feed the API reason codes.

## Limitations

- No calendar dates, only relative `DAYS_*` offsets, so there is no honest
  out-of-time back-test. Validation is a stratified holdout plus 5-fold
  stratified CV.
- The score is not a calibrated PD.
- Gender and family status are not features. Age and family size are, and
  they can correlate with protected groups. No disparate-impact audit was run.
- The loss proxy is illustrative. It is not an expected-loss or capital number.

## What changes with real telco and mobile-money data

- Split by calendar time and back-test on a later origination window.
- Add wallet, airtime, peer-to-peer, and merchant-payment behaviour with the
  same "known at application" rule used in the SQL.
- Keep a regularised scorecard beside the booster when a regulator or credit
  committee wants a points table.
- Calibrate the champion before using it as a PD, then monitor score PSI and
  feature CSI on weekly cohorts. Retrain or investigate when score PSI is at
  or above 0.25, or when two or more top drivers are in alert together.

## Notebook notes

| Notebook | Doc | What it covers |
| --- | --- | --- |
| `01_download_and_explore.ipynb` | `docs/01_download_and_explore.md` | Download and inventory |
| `02_data_quality_application.ipynb` | `docs/02_data_quality_application.md` | Application cleaning |
| `03_data_quality_child_tables.ipynb` | `docs/03_data_quality_child_tables.md` | Bureau, previous, instalments, POS, card |
| `04_feature_engineering.ipynb` | `docs/04_feature_engineering.md` | DuckDB feature build |
| `05_preprocess_and_splits.ipynb` | `docs/05_preprocess_and_splits.md` | Impute, cap, stratified splits |
| `06_linear_models.ipynb` | `docs/06_linear_models.md` | Logistic regression |
| `07_tree_models.ipynb` | `docs/07_tree_models.md` | Decision tree |
| `08_ensemble_models.ipynb` | `docs/08_ensemble_models.md` | Random forest, extra trees |
| `09_boosting_models.ipynb` | `docs/09_boosting_models.md` | XGBoost, LightGBM, CatBoost |
| `10_model_comparison.ipynb` | `docs/10_model_comparison.md` | Champion selection |
| `11_explainability.ipynb` | `docs/11_explainability.md` | SHAP |
| `12_monitoring_drift.ipynb` | `docs/12_monitoring_drift.md` | PSI / CSI |

## Repository layout

```
.
|-- api/                 FastAPI scoring service
|-- data/raw/            Kaggle CSVs (gitignored)
|-- data/processed/      Feature tables (gitignored)
|-- docs/                Cell-by-cell notebook notes
|-- notebooks/           Exploratory analysis
|-- reports/             Metrics, figures, model card
|-- sql/                 DuckDB feature SQL
|-- src/
|   |-- config.py        Paths and shared constants
|   |-- io_utils.py      Shared JSON / CSV helpers
|   |-- download_data.py
|   |-- data_quality/    Cleaning rules + DQ report
|   |-- features/        DuckDB SQL feature build
|   |-- preprocess/      Stratified splits + impute/cap pipeline
|   |-- modeling/        Training, metrics, plots, SHAP, registry
|   |-- train.py
|   |-- evaluate.py
|   |-- explain.py
|   |-- monitor.py
|   |-- scoring.py
|-- tests/
|-- Dockerfile
|-- Makefile
|-- requirements.txt
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
```

### Data

Download Home Credit Default Risk from Kaggle into `data/raw/`:

```bash
kaggle competitions download -c home-credit-default-risk -p data/raw
# unzip archives into data/raw/ so that application_train.csv etc. are present
```

Expected files under `data/raw/`:

- `application_train.csv`
- `bureau.csv`, `bureau_balance.csv`
- `previous_application.csv`
- `installments_payments.csv`
- `POS_CASH_balance.csv`
- `credit_card_balance.csv`

## Pipeline commands

```bash
make data        # download/unzip Home Credit raw tables
make quality     # data quality cleaning + reports/data_quality_report.md
make features    # DuckDB SQL feature build -> train_features.parquet
make preprocess  # stratified splits + train-only impute/cap pipeline
make train       # fit the champion on the full training split
make evaluate    # metrics -> reports/metrics.json and reports/model_card.md
make explain     # SHAP report
make monitor     # PSI / CSI drift report
make serve       # FastAPI on :8000
make test
```

`make evaluate` also writes the cutoff table and the age and income segment
AUCs into `reports/metrics.json`. `reports/model_card.md` restates intended
use, metrics, fairness notes, the monitoring trigger, and known risks.

## Feature data dictionary

One row per `SK_ID_CURR` in `data/processed/train_features.parquet`. Columns are
prefixed by source. All child aggregates use only relative offsets available at
application time (negative day/month offsets); see `sql/*.sql` and
`tests/test_feature_sql_leakage.py`.

| Prefix | Source | Example features |
| --- | --- | --- |
| `app_` | application_train | credit/income, annuity/income, credit/goods ratios; age and employment years; external scores; income, credit, annuity amounts |
| `inst_` | installments_payments | payment count, days-past-due mean/max, late rate, underpayment share, payment ratio, late-rate trend over last 3/6/12 months |
| `bur_` | bureau (+ bureau_balance) | active/closed credit counts, overdue amounts, max days overdue, utilisation, credit-history length, monthly DPD share |
| `prev_` | previous_application | approval/refusal rates, days since last application, mean requested vs approved amount |
| `pos_` | POS_CASH_balance | months observed, DPD mean/max, delinquency count, future-instalment trend |
| `cc_` | credit_card_balance | months observed, utilisation mean/max, max DPD, mean drawings, balance trend |

`TARGET` is the label (1 = payment difficulties). Count-style columns are 0 when
an applicant has no history in that source; other aggregates are NaN and imputed
in the preprocessing stage (fit on training folds only).

## Glossary

- **Default / TARGET**: the Home Credit label for payment difficulties.
- **DPD**: days past due on a scheduled payment.
- **Gini**: `2 * ROC AUC - 1`.
- **KS**: the largest gap between the cumulative distributions of goods and bads.
- **PD**: probability of default. This champion's output is a risk score, not a PD.
- **PSI / CSI**: population stability of the score, and of a single feature.

## Known dataset limitation

Home Credit exposes relative day offsets (`DAYS_*`), not calendar dates. A true
out-of-time back-test is not possible on this data alone. Reported validation
is a stratified holdout and stratified cross-validation.

## License / data

Code in this repository is original project work. Home Credit data remains
subject to Kaggle competition terms; raw files are not distributed here.
