"""Bureau / previous_application cleaning and SQL-only table notes (notebook 03)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DAYS_EMPLOYED_ANOMALY
from src.data_quality.decisions import Decision
from src.io_utils import replace_sentinel


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
        n_sent = int((prev[col] == DAYS_EMPLOYED_ANOMALY).sum())
        cleaned, flag = replace_sentinel(prev[col])
        prev[col] = cleaned
        prev[f"{col}_ANOMALY_FLAG"] = flag
        decisions.append(
            Decision(
                "previous_application",
                col,
                f"sentinel {DAYS_EMPLOYED_ANOMALY}",
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
