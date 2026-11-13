"""application_train cleaning rules (notebook 02)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DAYS_EMPLOYED_ANOMALY, ID_COL, TARGET_COL
from src.data_quality.decisions import Decision


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

    n_anom = int((clean["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY).sum())
    clean["DAYS_EMPLOYED_ANOMALY_FLAG"] = (clean["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY).astype("int8")
    clean.loc[clean["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY, "DAYS_EMPLOYED"] = np.nan
    clean["YEARS_EMPLOYED"] = (-clean["DAYS_EMPLOYED"] / 365.25).astype("float64")
    decisions.append(
        Decision(
            "application_train",
            "DAYS_EMPLOYED",
            f"sentinel {DAYS_EMPLOYED_ANOMALY}",
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
