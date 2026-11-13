"""Orchestrate Stage-2 data quality cleaning and report writing."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from src.config import (
    APPLICATION_CLEAN_PATH,
    APPLICATION_DECISIONS_PATH,
    APPLICATION_TRAIN_FILE,
    BUREAU_CLEAN_PATH,
    BUREAU_FILE,
    CHILD_DECISIONS_PATH,
    DATA_QUALITY_REPORT_PATH,
    DAYS_EMPLOYED_ANOMALY,
    ID_COL,
    PREVIOUS_APP_CLEAN_PATH,
    PREVIOUS_APPLICATION_FILE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RAW_TABLE_FILES,
    REPORTS_DIR,
)
from src.data_quality.application import clean_application
from src.data_quality.children import (
    clean_bureau,
    clean_previous_application,
    large_table_decisions,
)
from src.data_quality.report import build_report, join_key_coverage, table_row_counts
from src.io_utils import write_json


def run_data_quality() -> None:
    """Execute full Stage-2 quality pipeline and write artifacts."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    missing = [f for f in RAW_TABLE_FILES if not (RAW_DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing raw tables: {missing}. Run: python -m src.download_data"
        )

    print("Computing row counts ...", flush=True)
    counts = table_row_counts()

    print("Loading application_train ...", flush=True)
    app_path = RAW_DATA_DIR / APPLICATION_TRAIN_FILE
    app_raw = pd.read_csv(app_path, low_memory=False)
    app_clean, app_decisions = clean_application(app_raw)
    app_clean.to_parquet(APPLICATION_CLEAN_PATH, index=False)
    write_json(
        APPLICATION_DECISIONS_PATH,
        {
            "source": str(app_path),
            "n_rows_raw": int(len(app_raw)),
            "n_rows_clean": int(len(app_clean)),
            "n_cols_clean": int(app_clean.shape[1]),
            "decisions": [asdict(d) for d in app_decisions],
        },
    )
    print("Wrote", APPLICATION_CLEAN_PATH, flush=True)

    print("Loading bureau ...", flush=True)
    bureau_raw = pd.read_csv(RAW_DATA_DIR / BUREAU_FILE, low_memory=False)
    bureau_clean, bureau_decisions = clean_bureau(bureau_raw)
    bureau_clean.to_parquet(BUREAU_CLEAN_PATH, index=False)
    print("Wrote", BUREAU_CLEAN_PATH, flush=True)
    del bureau_raw, bureau_clean

    print("Loading previous_application ...", flush=True)
    prev_raw = pd.read_csv(RAW_DATA_DIR / PREVIOUS_APPLICATION_FILE, low_memory=False)
    prev_clean, prev_decisions = clean_previous_application(prev_raw)
    prev_clean.to_parquet(PREVIOUS_APP_CLEAN_PATH, index=False)
    print("Wrote", PREVIOUS_APP_CLEAN_PATH, flush=True)
    del prev_raw, prev_clean

    child_extra = large_table_decisions()
    all_decisions = app_decisions + bureau_decisions + prev_decisions + child_extra
    write_json(
        CHILD_DECISIONS_PATH,
        {
            "day_sentinel": int(DAYS_EMPLOYED_ANOMALY),
            "cleaned_parquets": [
                APPLICATION_CLEAN_PATH.name,
                BUREAU_CLEAN_PATH.name,
                PREVIOUS_APP_CLEAN_PATH.name,
            ],
            "sql_only_tables": [
                "bureau_balance.csv",
                "installments_payments.csv",
                "POS_CASH_balance.csv",
                "credit_card_balance.csv",
            ],
            "decisions": [
                asdict(d) for d in (bureau_decisions + prev_decisions + child_extra)
            ],
        },
    )

    print("Computing join-key coverage ...", flush=True)
    app_ids = set(app_clean[ID_COL].astype("int64"))
    coverage = join_key_coverage(app_ids)

    report = build_report(app_raw, app_clean, all_decisions, counts, coverage)
    DATA_QUALITY_REPORT_PATH.write_text(report, encoding="ascii", errors="replace")
    print("Wrote", DATA_QUALITY_REPORT_PATH, flush=True)
    print("Done. decisions=", len(all_decisions), flush=True)
