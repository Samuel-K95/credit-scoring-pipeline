"""Schema contract for the modelling table built by sql/05_build_training_table.sql.

The SQL text is the always-on check. When train_features.parquet is present,
the written table must match that contract: one row per applicant, binary
target, and the prefixed feature columns.
"""

from __future__ import annotations

import pytest

from src.config import ID_COL, SQL_DIR, TARGET_COL, TRAIN_FEATURES_PATH

# Column order follows the SELECT in sql/05_build_training_table.sql.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "SK_ID_CURR",
    "TARGET",
    "app_credit_to_income",
    "app_annuity_to_income",
    "app_credit_to_goods",
    "app_age_years",
    "app_years_employed",
    "app_amt_income_total",
    "app_amt_credit",
    "app_amt_annuity",
    "app_ext_source_1",
    "app_ext_source_2",
    "app_ext_source_3",
    "app_cnt_children",
    "app_cnt_fam_members",
    "inst_count",
    "inst_dpd_mean",
    "inst_dpd_max",
    "inst_late_rate",
    "inst_underpay_share",
    "inst_pay_ratio_mean",
    "inst_late_rate_3m",
    "inst_late_rate_6m",
    "inst_late_rate_12m",
    "bur_count",
    "bur_active_count",
    "bur_closed_count",
    "bur_overdue_amt_sum",
    "bur_max_overdue_amt",
    "bur_max_days_overdue",
    "bur_utilisation",
    "bur_history_length_days",
    "bur_amt_credit_sum_mean",
    "bur_bb_dpd_share_mean",
    "bur_bb_months_total",
    "prev_count",
    "prev_approved_rate",
    "prev_refused_rate",
    "prev_days_since_last",
    "prev_amt_application_mean",
    "prev_amt_credit_mean",
    "prev_credit_to_application_mean",
    "pos_months_count",
    "pos_dpd_mean",
    "pos_dpd_max",
    "pos_dpd_def_max",
    "pos_delinquency_count",
    "pos_future_inst_trend",
    "cc_months_count",
    "cc_utilisation_mean",
    "cc_utilisation_max",
    "cc_dpd_max",
    "cc_drawings_mean",
    "cc_balance_trend",
)


def test_training_sql_declares_expected_columns() -> None:
    sql = (SQL_DIR / "05_build_training_table.sql").read_text(encoding="ascii")
    for col in EXPECTED_COLUMNS:
        assert col in sql, f"training SQL missing column {col}"
    assert ID_COL in EXPECTED_COLUMNS
    assert TARGET_COL in EXPECTED_COLUMNS


def test_feature_prefixes_cover_each_source() -> None:
    prefixes = ("app_", "inst_", "bur_", "prev_", "pos_", "cc_")
    features = [c for c in EXPECTED_COLUMNS if c not in {ID_COL, TARGET_COL}]
    assert len(features) == 52
    for prefix in prefixes:
        assert any(c.startswith(prefix) for c in features)


@pytest.mark.skipif(
    not TRAIN_FEATURES_PATH.exists(),
    reason="train_features.parquet not built yet",
)
def test_parquet_matches_training_schema() -> None:
    import pyarrow.parquet as pq

    # Schema plus the two key columns only. Reading every feature column
    # is unnecessary and has blown the memory budget on this machine.
    pf = pq.ParquetFile(TRAIN_FEATURES_PATH)
    assert list(pf.schema_arrow.names) == list(EXPECTED_COLUMNS)
    assert pf.metadata.num_rows > 0
    keys = pq.read_table(TRAIN_FEATURES_PATH, columns=[ID_COL, TARGET_COL]).to_pandas()
    assert keys[ID_COL].is_unique
    assert set(keys[TARGET_COL].dropna().unique()).issubset({0, 1})
