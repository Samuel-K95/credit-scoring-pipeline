"""No-leakage checks on feature SQL: no future (positive) time offsets used.

Home Credit relative offsets (DAYS_*, MONTHS_BALANCE) must be <= 0 at
application time. Each child aggregation must filter on them accordingly and
must not select records with positive offsets.
"""

from __future__ import annotations

import re

from src.config import SQL_DIR

# Time-offset columns that must never be used with a future (> 0 / >= positive) filter.
OFFSET_COLS = (
    "DAYS_CREDIT",
    "DAYS_DECISION",
    "DAYS_INSTALMENT",
    "MONTHS_BALANCE",
)

CHILD_SQL_FILES = (
    "01_bureau_features.sql",
    "02_installments_features.sql",
    "03_previous_app_features.sql",
    "04_pos_cc_features.sql",
)


def _read(name: str) -> str:
    return (SQL_DIR / name).read_text(encoding="ascii")


def test_each_child_sql_filters_offsets_to_non_positive() -> None:
    """Every child feature file must constrain its driving offset to <= 0."""
    expected = {
        "01_bureau_features.sql": ["DAYS_CREDIT <= 0", "MONTHS_BALANCE <= 0"],
        "02_installments_features.sql": ["DAYS_INSTALMENT <= 0"],
        "03_previous_app_features.sql": ["DAYS_DECISION <= 0"],
        "04_pos_cc_features.sql": ["MONTHS_BALANCE <= 0"],
    }
    for name, needles in expected.items():
        sql = _read(name)
        for needle in needles:
            assert needle in sql, f"{name} missing leakage guard: {needle}"


def test_no_positive_offset_filters() -> None:
    """No SQL should filter an offset column to strictly positive values."""
    for name in CHILD_SQL_FILES:
        sql = _read(name)
        for col in OFFSET_COLS:
            # Flag patterns like "DAYS_CREDIT > 0" or "MONTHS_BALANCE >= 1".
            bad = re.search(rf"{col}\s*>=?\s*0*[1-9]", sql) or re.search(
                rf"{col}\s*>\s*0", sql
            )
            assert bad is None, f"{name}: possible future-offset filter on {col}"


def test_trend_windows_use_negative_bounds() -> None:
    """Installment trend windows must use negative day bounds (past only)."""
    sql = _read("02_installments_features.sql")
    for bound in ("-90", "-180", "-365"):
        assert f">= {bound}" in sql, f"installments trend window missing {bound}"
