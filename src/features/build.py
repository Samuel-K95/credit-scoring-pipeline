"""Run DuckDB SQL feature scripts and write train_features.parquet.

Ported from notebooks/04_feature_engineering.ipynb. Registers the cleaned
Stage-2 outputs and raw child CSVs as DuckDB views, then executes the ordered
SQL files in sql/. All child aggregations filter to relative offsets available
at application time (<= 0); see the per-file leakage guards and tests.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from src.config import (
    APPLICATION_CLEAN_PATH,
    BUREAU_BALANCE_FILE,
    BUREAU_CLEAN_PATH,
    CREDIT_CARD_BALANCE_FILE,
    ID_COL,
    INSTALLMENTS_PAYMENTS_FILE,
    POS_CASH_BALANCE_FILE,
    PREVIOUS_APP_CLEAN_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SQL_DIR,
    TRAIN_FEATURES_PATH,
)

SQL_FILES: tuple[str, ...] = (
    "01_bureau_features.sql",
    "02_installments_features.sql",
    "03_previous_app_features.sql",
    "04_pos_cc_features.sql",
    "05_build_training_table.sql",
)


def register_views(con: duckdb.DuckDBPyConnection) -> None:
    """Create DuckDB views the SQL files reference by name."""
    app = APPLICATION_CLEAN_PATH.as_posix()
    bureau = BUREAU_CLEAN_PATH.as_posix()
    prev = PREVIOUS_APP_CLEAN_PATH.as_posix()
    bb = (RAW_DATA_DIR / BUREAU_BALANCE_FILE).as_posix()
    inst = (RAW_DATA_DIR / INSTALLMENTS_PAYMENTS_FILE).as_posix()
    pos = (RAW_DATA_DIR / POS_CASH_BALANCE_FILE).as_posix()
    cc = (RAW_DATA_DIR / CREDIT_CARD_BALANCE_FILE).as_posix()

    con.execute(f"CREATE OR REPLACE VIEW app AS SELECT * FROM read_parquet('{app}')")
    con.execute(
        f"CREATE OR REPLACE VIEW bureau AS SELECT * FROM read_parquet('{bureau}')"
    )
    con.execute(
        f"CREATE OR REPLACE VIEW previous_application AS SELECT * FROM read_parquet('{prev}')"
    )
    con.execute(
        f"CREATE OR REPLACE VIEW bureau_balance AS SELECT * FROM read_csv_auto('{bb}')"
    )
    con.execute(
        f"CREATE OR REPLACE VIEW installments AS SELECT * FROM read_csv_auto('{inst}')"
    )
    con.execute(
        f"CREATE OR REPLACE VIEW pos_cash AS SELECT * FROM read_csv_auto('{pos}')"
    )
    con.execute(
        f"CREATE OR REPLACE VIEW credit_card AS SELECT * FROM read_csv_auto('{cc}')"
    )


def build_features(output_path: Path = TRAIN_FEATURES_PATH) -> Path:
    """Execute the SQL pipeline and write the joined feature table to parquet."""
    missing_inputs = [
        p
        for p in (APPLICATION_CLEAN_PATH, BUREAU_CLEAN_PATH, PREVIOUS_APP_CLEAN_PATH)
        if not p.exists()
    ]
    if missing_inputs:
        raise FileNotFoundError(
            "Missing cleaned inputs: "
            + ", ".join(str(p) for p in missing_inputs)
            + ". Run: python -m src.data_quality"
        )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute("PRAGMA threads=4")
        register_views(con)
        for name in SQL_FILES:
            sql = (SQL_DIR / name).read_text(encoding="ascii")
            print(f"Running {name} ...", flush=True)
            con.execute(sql)

        n_rows = con.execute("SELECT COUNT(*) FROM train_features").fetchone()[0]
        n_dupes = con.execute(
            f"SELECT COUNT(*) - COUNT(DISTINCT {ID_COL}) FROM train_features"
        ).fetchone()[0]
        if n_dupes != 0:
            raise ValueError(f"train_features has {n_dupes} duplicate {ID_COL} rows")

        con.execute(
            f"COPY train_features TO '{output_path.as_posix()}' (FORMAT PARQUET)"
        )
        n_cols = len(con.execute("SELECT * FROM train_features LIMIT 0").df().columns)
    finally:
        con.close()

    print(f"Wrote {output_path} ({n_rows} rows, {n_cols} cols)", flush=True)
    return output_path
