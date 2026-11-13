# Credit Scoring Pipeline

Production-style credit scoring project for digital lending: application-time
probability of default (PD), scorecard and gradient-boosted models, validation,
monitoring, and a FastAPI scoring service.

## Problem framing

A telecom and mobile-money lender wants to automate approve/decline and limit
decisions for digital loans. The model estimates PD at application time using
repayment history, transactional behaviour, and bureau-style attributes.

This repository uses the public **Home Credit Default Risk** dataset from Kaggle
as a stand-in for proprietary lending data. Home Credit is consumer credit data,
not telecom or mobile-money transaction data.

## Status

Stages 1-4 are implemented (download, data quality, DuckDB features,
preprocess/splits). Stage 5 modelling is under way: notebooks 06-09 explore
linear, tree, bagging-ensemble, and boosting models with a shared
`src.modeling` toolkit (reusable training, GridSearchCV tuning, and train vs
validation curves). The final champion will be ported to `src/train.py`.
Reported metrics will come only from `reports/metrics.json` after the pipeline
trains the chosen model.

## Modelling notebooks (Stage 5)

| Notebook | Family | Models |
| --- | --- | --- |
| `06_linear_models.ipynb` | Linear | Logistic regression (scorecard base) |
| `07_tree_models.ipynb` | Tree | Decision tree (CART) |
| `08_ensemble_models.ipynb` | Ensemble (bagging) | Random forest, extra trees |
| `09_boosting_models.ipynb` | Boosting | XGBoost, LightGBM, CatBoost, GradientBoosting |

Each notebook tunes with `GridSearchCV` over the saved stratified folds, plots
validation/learning curves and train-vs-validation loss curves, and evaluates
the selected model once on the untouched holdout. Preprocessing is refit inside
every fold via `src.modeling.make_pipeline`, so there is no leakage.

## Repository layout

```
.
|-- api/                 FastAPI scoring service
|-- data/raw/            Kaggle CSVs (gitignored)
|-- data/processed/      Feature tables (gitignored)
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
|   |-- modeling/        Stage 5 toolkit (data, metrics, plots, search, curves)
|   |-- train.py         (Stage 5)
|   |-- evaluate.py      (Stage 6)
|   |-- explain.py       (Stage 7)
|   |-- monitor.py       (Stage 8)
|   |-- scoring.py       Scorecard helpers
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
make train       # scorecard + LightGBM
make evaluate    # metrics -> reports/metrics.json
make monitor     # PSI / CSI drift report
make serve       # FastAPI on :8000
make test
```

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

## Known dataset limitation

Home Credit exposes relative day offsets (`DAYS_*`), not calendar dates. A true
out-of-time back-test is not possible on this data alone. Validation will use a
stratified holdout and stratified cross-validation, documented clearly when
results are reported.

## License / data

Code in this repository is original project work. Home Credit data remains
subject to Kaggle competition terms; raw files are not distributed here.
