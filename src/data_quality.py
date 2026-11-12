"""Data quality checks, cleaning, and markdown report.

Ported from:
- notebooks/02_data_quality_application.ipynb
- notebooks/03_data_quality_child_tables.ipynb
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.config import (
    APPLICATION_CLEAN_PATH,
    APPLICATION_DECISIONS_PATH,
    APPLICATION_TRAIN_FILE,
    BUREAU_CLEAN_PATH,
    BUREAU_FILE,
    CHILD_DECISIONS_PATH,
    CREDIT_CARD_BALANCE_FILE,
    CSV_CHUNKSIZE,
    DATA_QUALITY_REPORT_PATH,
    DAYS_EMPLOYED_ANOMALY,
    ID_COL,
    INSTALLMENTS_PAYMENTS_FILE,
    POS_CASH_BALANCE_FILE,
    PREVIOUS_APP_CLEAN_PATH,
    PREVIOUS_APPLICATION_FILE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RAW_TABLE_FILES,
    REPORTS_DIR,
    TARGET_COL,
)

DAY_SENTINEL = DAYS_EMPLOYED_ANOMALY


@dataclass
class Decision:
    """One documented cleaning / quality decision."""

    table: str
    feature: str
    issue: str
    action: str
    detail: str = ""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="ascii", errors="replace")


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table without requiring the tabulate package."""
    if df.empty:
        return "_empty_"
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        cells = ["" if pd.isna(v) else str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _unique_ids(path: Path, id_col: str) -> set[int]:
    ids: set[int] = set()
    for chunk in pd.read_csv(path, usecols=[id_col], chunksize=CSV_CHUNKSIZE):
        vals = chunk[id_col].dropna().astype("int64").unique()
        ids.update(vals.tolist())
    return ids


def _replace_sentinel(series: pd.Series, sentinel: int = DAY_SENTINEL) -> tuple[pd.Series, pd.Series]:
    flag = (series == sentinel).astype("int8")
    cleaned = series.mask(series == sentinel, np.nan)
    return cleaned, flag


# ---------------------------------------------------------------------------
# application_train cleaning
# ---------------------------------------------------------------------------


def clean_application(df: pd.DataFrame) -> tuple[pd.DataFrame, list[Decision]]:
    """Apply notebook-02 cleaning rules to application_train."""
    decisions: list[Decision] = []
    clean = df.copy()

    decisions.append(
        Decision("application_train", TARGET_COL, "none", "keep as-is", "binary label")
    )
    decisions.append(
        Decision("application_train", ID_COL, "none", "keep as-is", "unique primary key")
    )

    n_xna = int((clean["CODE_GENDER"] == "XNA").sum())
    clean = clean.loc[clean["CODE_GENDER"] != "XNA"].copy()
    decisions.append(
        Decision(
            "application_train",
            "CODE_GENDER",
            "XNA placeholder",
            "drop rows where CODE_GENDER == XNA",
            f"removed={n_xna}",
        )
    )

    for col, fill in [
        ("NAME_TYPE_SUITE", "Missing"),
        ("OCCUPATION_TYPE", "Missing"),
    ]:
        n_miss = int(clean[col].isna().sum())
        clean[col] = clean[col].fillna(fill)
        decisions.append(
            Decision(
                "application_train",
                col,
                "categorical missing",
                f"fillna '{fill}'",
                f"n_miss={n_miss}",
            )
        )

    n_unknown = int((clean["NAME_FAMILY_STATUS"] == "Unknown").sum())
    clean.loc[clean["NAME_FAMILY_STATUS"] == "Unknown", "NAME_FAMILY_STATUS"] = "Missing"
    decisions.append(
        Decision(
            "application_train",
            "NAME_FAMILY_STATUS",
            "Unknown placeholder",
            "map Unknown -> Missing",
            f"n={n_unknown}",
        )
    )

    n_anom = int((clean["DAYS_EMPLOYED"] == DAY_SENTINEL).sum())
    clean["DAYS_EMPLOYED_ANOMALY_FLAG"] = (clean["DAYS_EMPLOYED"] == DAY_SENTINEL).astype("int8")
    clean.loc[clean["DAYS_EMPLOYED"] == DAY_SENTINEL, "DAYS_EMPLOYED"] = np.nan
    clean["YEARS_EMPLOYED"] = (-clean["DAYS_EMPLOYED"] / 365.25).astype("float64")
    decisions.append(
        Decision(
            "application_train",
            "DAYS_EMPLOYED",
            f"sentinel {DAY_SENTINEL}",
            "set anomaly to NaN; add flag; derive YEARS_EMPLOYED",
            f"n_anom={n_anom}",
        )
    )

    clean["AGE_YEARS"] = (-clean["DAYS_BIRTH"] / 365.25).astype("float64")
    clean["YEARS_REGISTRATION"] = (-clean["DAYS_REGISTRATION"] / 365.25).astype("float64")
    clean["YEARS_ID_PUBLISH"] = (-clean["DAYS_ID_PUBLISH"] / 365.25).astype("float64")
    decisions.append(
        Decision(
            "application_train",
            "DAYS_BIRTH",
            "relative offset",
            "derive AGE_YEARS",
        )
    )
    decisions.append(
        Decision(
            "application_train",
            "DAYS_REGISTRATION",
            "relative offset",
            "derive YEARS_REGISTRATION",
        )
    )
    decisions.append(
        Decision(
            "application_train",
            "DAYS_ID_PUBLISH",
            "relative offset",
            "derive YEARS_ID_PUBLISH",
        )
    )

    for col in [
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "OWN_CAR_AGE",
        "CNT_FAM_MEMBERS",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]:
        if col not in clean.columns:
            continue
        n_miss = int(clean[col].isna().sum())
        clean[f"{col}_MISSING"] = clean[col].isna().astype("int8")
        decisions.append(
            Decision(
                "application_train",
                col,
                f"{n_miss} missing",
                f"keep NaN + {col}_MISSING",
            )
        )

    n_org = int((clean["ORGANIZATION_TYPE"] == "XNA").sum())
    clean["ORGANIZATION_TYPE"] = clean["ORGANIZATION_TYPE"].replace({"XNA": "Missing"})
    decisions.append(
        Decision(
            "application_train",
            "ORGANIZATION_TYPE",
            "XNA placeholder",
            "map XNA -> Missing",
            f"n_xna={n_org}",
        )
    )

    housing_cats = [
        "FONDKAPREMONT_MODE",
        "HOUSETYPE_MODE",
        "WALLSMATERIAL_MODE",
        "EMERGENCYSTATE_MODE",
    ]
    for col in housing_cats:
        if col not in clean.columns:
            continue
        n_miss = int(clean[col].isna().sum())
        clean[col] = clean[col].fillna("Missing")
        decisions.append(
            Decision(
                "application_train",
                col,
                f"{n_miss} missing categorical",
                "fillna 'Missing'",
            )
        )

    housing_nums = [
        c
        for c in clean.columns
        if c.endswith(("_AVG", "_MODE", "_MEDI")) or c == "TOTALAREA_MODE"
    ]
    housing_nums = [c for c in housing_nums if c not in housing_cats and pd.api.types.is_numeric_dtype(clean[c])]
    for col in housing_nums:
        n_miss = int(clean[col].isna().sum())
        clean[f"{col}_MISSING"] = clean[col].isna().astype("int8")
        decisions.append(
            Decision(
                "application_train",
                col,
                f"{n_miss} missing",
                f"keep NaN + {col}_MISSING",
            )
        )

    clean["app_credit_to_income"] = clean["AMT_CREDIT"] / clean["AMT_INCOME_TOTAL"].replace(0, np.nan)
    clean["app_annuity_to_income"] = clean["AMT_ANNUITY"] / clean["AMT_INCOME_TOTAL"].replace(0, np.nan)
    clean["app_credit_to_goods"] = clean["AMT_CREDIT"] / clean["AMT_GOODS_PRICE"].replace(0, np.nan)
    decisions.append(
        Decision(
            "application_train",
            "app_ratios",
            "derived",
            "add credit/income, annuity/income, credit/goods ratios",
        )
    )

    clean["AMT_INCOME_TOTAL_LOG1P"] = np.log1p(clean["AMT_INCOME_TOTAL"])
    clean["AMT_CREDIT_LOG1P"] = np.log1p(clean["AMT_CREDIT"])

    return clean, decisions


# ---------------------------------------------------------------------------
# bureau / previous_application cleaning
# ---------------------------------------------------------------------------


def clean_bureau(df: pd.DataFrame) -> tuple[pd.DataFrame, list[Decision]]:
    """Apply notebook-03 cleaning rules to bureau."""
    decisions: list[Decision] = []
    bureau = df.copy()

    for col in [
        "DAYS_CREDIT_ENDDATE",
        "DAYS_ENDDATE_FACT",
        "AMT_CREDIT_MAX_OVERDUE",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_LIMIT",
        "AMT_ANNUITY",
    ]:
        if col not in bureau.columns:
            continue
        if bureau[col].isna().any():
            bureau[f"{col}_MISSING"] = bureau[col].isna().astype("int8")
            decisions.append(
                Decision("bureau", col, "missing values", f"keep NaN + {col}_MISSING")
            )

    before = bureau["DAYS_CREDIT_UPDATE"].copy()
    n_pos = int((before > 0).sum())
    bureau.loc[bureau["DAYS_CREDIT_UPDATE"] > 0, "DAYS_CREDIT_UPDATE"] = np.nan
    bureau["DAYS_CREDIT_UPDATE_POS_FLAG"] = (before > 0).astype("int8")
    decisions.append(
        Decision(
            "bureau",
            "DAYS_CREDIT_UPDATE",
            f"{n_pos} positive values",
            "set >0 to NaN + flag",
        )
    )
    decisions.append(
        Decision(
            "bureau",
            "DAYS_CREDIT",
            "relative offset",
            "keep; require <= 0 for application-time features",
        )
    )
    return bureau, decisions


def clean_previous_application(df: pd.DataFrame) -> tuple[pd.DataFrame, list[Decision]]:
    """Apply notebook-03 cleaning rules to previous_application."""
    decisions: list[Decision] = []
    prev = df.copy()

    day_cols = [
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]
    for col in day_cols:
        if col not in prev.columns:
            continue
        n_sent = int((prev[col] == DAY_SENTINEL).sum())
        cleaned, flag = _replace_sentinel(prev[col])
        prev[col] = cleaned
        prev[f"{col}_ANOMALY_FLAG"] = flag
        decisions.append(
            Decision(
                "previous_application",
                col,
                f"sentinel {DAY_SENTINEL}",
                f"set sentinel to NaN + {col}_ANOMALY_FLAG",
                f"n={n_sent}",
            )
        )

    cat_cols = [
        "NAME_CONTRACT_TYPE",
        "NAME_CASH_LOAN_PURPOSE",
        "CODE_REJECT_REASON",
        "NAME_CLIENT_TYPE",
        "NAME_GOODS_CATEGORY",
        "NAME_PORTFOLIO",
        "NAME_PRODUCT_TYPE",
        "CHANNEL_TYPE",
        "NAME_SELLER_INDUSTRY",
        "NAME_YIELD_GROUP",
        "PRODUCT_COMBINATION",
        "NAME_PAYMENT_TYPE",
        "NAME_TYPE_SUITE",
    ]
    for col in cat_cols:
        if col not in prev.columns:
            continue
        n_x = int(prev[col].isin(["XNA", "XAP"]).sum())
        n_miss = int(prev[col].isna().sum())
        prev[col] = prev[col].replace({"XNA": "Missing", "XAP": "Missing"}).fillna("Missing")
        decisions.append(
            Decision(
                "previous_application",
                col,
                f"XNA/XAP/NaN n_x={n_x} n_miss={n_miss}",
                "map XNA/XAP/NaN -> Missing",
            )
        )

    for col in [
        "AMT_ANNUITY",
        "AMT_DOWN_PAYMENT",
        "AMT_GOODS_PRICE",
        "RATE_DOWN_PAYMENT",
        "RATE_INTEREST_PRIMARY",
        "RATE_INTEREST_PRIVILEGED",
        "CNT_PAYMENT",
    ]:
        if col not in prev.columns:
            continue
        if prev[col].isna().any():
            prev[f"{col}_MISSING"] = prev[col].isna().astype("int8")
            decisions.append(
                Decision(
                    "previous_application",
                    col,
                    "missing",
                    f"keep NaN + {col}_MISSING",
                )
            )

    if "SELLERPLACE_AREA" in prev.columns:
        n_m1 = int((prev["SELLERPLACE_AREA"] == -1).sum())
        prev.loc[prev["SELLERPLACE_AREA"] == -1, "SELLERPLACE_AREA"] = np.nan
        prev["SELLERPLACE_AREA_MISSING"] = prev["SELLERPLACE_AREA"].isna().astype("int8")
        decisions.append(
            Decision(
                "previous_application",
                "SELLERPLACE_AREA",
                f"-1 unknown n={n_m1}",
                "set -1 to NaN + missing flag",
            )
        )

    decisions.append(
        Decision(
            "previous_application",
            "DAYS_DECISION",
            "relative offset",
            "keep; filter DAYS_DECISION <= 0 for application-time features",
        )
    )
    return prev, decisions


def large_table_decisions() -> list[Decision]:
    """Document rules for tables cleaned in SQL rather than rewritten to parquet."""
    return [
        Decision(
            "bureau_balance",
            "MONTHS_BALANCE",
            "relative month offset",
            "use only MONTHS_BALANCE <= 0 at feature time",
        ),
        Decision(
            "bureau_balance",
            "STATUS",
            "none",
            "keep; C=closed, X=unknown, 0-5 DPD buckets",
        ),
        Decision(
            "installments_payments",
            "DAYS_ENTRY_PAYMENT",
            "missing when payment timing unknown",
            "keep NaN; derive late/underpay only when both day fields present",
        ),
        Decision(
            "installments_payments",
            "DAYS_INSTALMENT",
            "relative offset",
            "use only offsets available at application time in SQL",
        ),
        Decision(
            "POS_CASH_balance",
            "MONTHS_BALANCE",
            "relative month",
            "use only <= 0",
        ),
        Decision(
            "POS_CASH_balance",
            "SK_DPD",
            "delinquency days",
            "keep; aggregate in SQL",
        ),
        Decision(
            "credit_card_balance",
            "AMT_RECIVABLE",
            "dataset typo in name",
            "keep original column name",
        ),
        Decision(
            "credit_card_balance",
            "MONTHS_BALANCE",
            "relative month",
            "use only <= 0",
        ),
        Decision(
            "credit_card_balance",
            "utilisation",
            "limit can be 0",
            "safe-divide in SQL",
        ),
    ]


# ---------------------------------------------------------------------------
# Profiling helpers for the report
# ---------------------------------------------------------------------------


def _count_csv_rows(path: Path) -> int:
    """Count data rows (excludes header) without parsing CSV fields."""
    with path.open("rb") as f:
        n = sum(1 for _ in f)
    return max(n - 1, 0)


def table_row_counts() -> pd.DataFrame:
    """Row counts and on-disk sizes for raw tables."""
    rows: list[dict[str, Any]] = []
    for name in RAW_TABLE_FILES:
        path = RAW_DATA_DIR / name
        if not path.exists():
            rows.append({"table": name, "rows": None, "size_mb": None, "status": "missing"})
            continue
        print(f"  counting {name} ...", flush=True)
        n_rows = _count_csv_rows(path)
        rows.append(
            {
                "table": name,
                "rows": int(n_rows),
                "size_mb": round(path.stat().st_size / (1024**2), 2),
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def join_key_coverage(app_ids: set[int]) -> pd.DataFrame:
    """Share of application IDs present in each child table."""
    specs = [
        (BUREAU_FILE, ID_COL),
        (PREVIOUS_APPLICATION_FILE, ID_COL),
        (INSTALLMENTS_PAYMENTS_FILE, ID_COL),
        (POS_CASH_BALANCE_FILE, ID_COL),
        (CREDIT_CARD_BALANCE_FILE, ID_COL),
    ]
    rows: list[dict[str, Any]] = []
    for name, id_col in specs:
        path = RAW_DATA_DIR / name
        if not path.exists():
            rows.append({"table": name, "status": "missing"})
            continue
        print(f"  coverage for {name} ...", flush=True)
        child_ids = _unique_ids(path, id_col)
        overlap = app_ids & child_ids
        rows.append(
            {
                "table": name,
                "status": "ok",
                "n_ids_in_child": len(child_ids),
                "n_overlap_with_app": len(overlap),
                "pct_app_with_child": round(100 * len(overlap) / max(len(app_ids), 1), 2),
            }
        )
    return pd.DataFrame(rows)


def build_report(
    app_raw: pd.DataFrame,
    app_clean: pd.DataFrame,
    decisions: Iterable[Decision],
    counts: pd.DataFrame,
    coverage: pd.DataFrame,
) -> str:
    """Render reports/data_quality_report.md content."""
    lines: list[str] = []
    lines.append("# Data quality report")
    lines.append("")
    lines.append("Generated by `src/data_quality.py` from Home Credit raw tables.")
    lines.append("")
    lines.append("## Row counts per table")
    lines.append("")
    lines.append(_df_to_markdown(counts))
    lines.append("")

    target_rate = float(app_clean[TARGET_COL].mean())
    n_dup = int(app_clean[ID_COL].duplicated().sum())
    n_anom_raw = int((app_raw["DAYS_EMPLOYED"] == DAY_SENTINEL).sum())
    lines.append("## application_train target and keys")
    lines.append("")
    lines.append(f"- Rows (raw): {len(app_raw):,}")
    lines.append(f"- Rows (clean): {len(app_clean):,}")
    lines.append(f"- Target rate: {target_rate:.4f}")
    lines.append(f"- Duplicate `{ID_COL}` after clean: {n_dup}")
    lines.append(
        f"- `DAYS_EMPLOYED == {DAY_SENTINEL}` in raw: {n_anom_raw:,} "
        f"({100 * n_anom_raw / max(len(app_raw), 1):.2f}%)"
    )
    lines.append("")

    lines.append("## Join-key coverage vs application_train")
    lines.append("")
    lines.append(_df_to_markdown(coverage))
    lines.append("")

    lines.append("## Top missingness (raw application_train)")
    lines.append("")
    miss_app = (
        (app_raw.isna().mean() * 100)
        .sort_values(ascending=False)
        .head(20)
        .rename("pct_missing")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    lines.append(_df_to_markdown(miss_app))
    lines.append("")

    lines.append("## Cleaning decisions")
    lines.append("")
    dec_df = pd.DataFrame([asdict(d) for d in decisions])
    lines.append(_df_to_markdown(dec_df.head(80)))
    if len(dec_df) > 80:
        lines.append("")
        lines.append(f"_Showing 80 / {len(dec_df)} decisions. Full log in processed JSON._")
    lines.append("")

    lines.append("## Known dataset limitations")
    lines.append("")
    lines.append(
        "- Home Credit uses relative day/month offsets, not calendar dates; "
        "true out-of-time validation is not possible on this dataset alone."
    )
    lines.append(
        f"- Day sentinel `{DAY_SENTINEL}` appears in employment and previous-application day fields."
    )
    lines.append(
        "- Large child tables (`bureau_balance`, `installments_payments`, "
        "`POS_CASH_balance`, `credit_card_balance`) are profiled here; "
        "row-level cleaning is applied in DuckDB feature SQL."
    )
    lines.append("")
    return "\n".join(lines)


def run_data_quality() -> None:
    """Execute full Stage-2 quality pipeline and write artifacts."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    missing = [f for f in RAW_TABLE_FILES if not (RAW_DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw tables: {missing}. Run: python -m src.download_data")

    print("Computing row counts ...", flush=True)
    counts = table_row_counts()

    print("Loading application_train ...", flush=True)
    app_path = RAW_DATA_DIR / APPLICATION_TRAIN_FILE
    app_raw = pd.read_csv(app_path, low_memory=False)
    app_clean, app_decisions = clean_application(app_raw)
    app_clean.to_parquet(APPLICATION_CLEAN_PATH, index=False)
    _write_json(
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
    _write_json(
        CHILD_DECISIONS_PATH,
        {
            "day_sentinel": int(DAY_SENTINEL),
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
            "decisions": [asdict(d) for d in (bureau_decisions + prev_decisions + child_extra)],
        },
    )

    print("Computing join-key coverage ...", flush=True)
    app_ids = set(app_clean[ID_COL].astype("int64"))
    coverage = join_key_coverage(app_ids)

    report = build_report(app_raw, app_clean, all_decisions, counts, coverage)
    DATA_QUALITY_REPORT_PATH.write_text(report, encoding="ascii", errors="replace")
    print("Wrote", DATA_QUALITY_REPORT_PATH, flush=True)
    print("Done. decisions=", len(all_decisions), flush=True)


def main() -> None:
    """CLI entrypoint."""
    run_data_quality()


if __name__ == "__main__":
    main()
