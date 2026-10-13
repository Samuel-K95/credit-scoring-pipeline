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

Scaffold and project layout are in place. Pipeline stages (data quality,
features, training, evaluation, monitoring, API) will be filled in as work
progresses. Reported metrics will come only from `reports/metrics.json` after
models are actually trained.

## Repository layout

```
.
|-- api/                 FastAPI scoring service
|-- data/raw/            Kaggle CSVs (gitignored)
|-- data/processed/      Feature tables (gitignored)
|-- notebooks/           Exploratory analysis
|-- reports/             Metrics, figures, model card
|-- sql/                 DuckDB feature SQL
|-- src/                 Pipeline modules
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
make data        # check raw tables are present
make features    # DuckDB SQL feature build
make train       # scorecard + LightGBM
make evaluate    # metrics -> reports/metrics.json
make monitor     # PSI / CSI drift report
make serve       # FastAPI on :8000
make test
```

## Known dataset limitation

Home Credit exposes relative day offsets (`DAYS_*`), not calendar dates. A true
out-of-time back-test is not possible on this data alone. Validation will use a
stratified holdout and stratified cross-validation, documented clearly when
results are reported.

## License / data

Code in this repository is original project work. Home Credit data remains
subject to Kaggle competition terms; raw files are not distributed here.
