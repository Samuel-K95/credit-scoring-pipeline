-- POS cash and credit card balance aggregates to one row per SK_ID_CURR.
-- Prefixes: pos_, cc_
-- Leakage guard: only months at/before application (MONTHS_BALANCE <= 0).
-- Note: SK_DPD > 0 is a delinquency flag, not a time offset.
-- Depends on views: pos_cash, credit_card.
CREATE OR REPLACE TABLE pos_features AS
WITH pos AS (
    SELECT * FROM pos_cash WHERE MONTHS_BALANCE <= 0
)
SELECT
    SK_ID_CURR,
    COUNT(*)                                             AS pos_months_count,
    AVG(SK_DPD)                                          AS pos_dpd_mean,
    MAX(SK_DPD)                                          AS pos_dpd_max,
    MAX(SK_DPD_DEF)                                      AS pos_dpd_def_max,
    SUM(CASE WHEN SK_DPD > 0 THEN 1 ELSE 0 END)          AS pos_delinquency_count,
    (arg_max(CNT_INSTALMENT_FUTURE, MONTHS_BALANCE)
        - arg_min(CNT_INSTALMENT_FUTURE, MONTHS_BALANCE)) AS pos_future_inst_trend
FROM pos
GROUP BY SK_ID_CURR;

CREATE OR REPLACE TABLE cc_features AS
WITH cc AS (
    SELECT
        *,
        AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0) AS util
    FROM credit_card
    WHERE MONTHS_BALANCE <= 0
)
SELECT
    SK_ID_CURR,
    COUNT(*)                                             AS cc_months_count,
    AVG(util)                                            AS cc_utilisation_mean,
    MAX(util)                                            AS cc_utilisation_max,
    MAX(SK_DPD)                                          AS cc_dpd_max,
    AVG(AMT_DRAWINGS_CURRENT)                            AS cc_drawings_mean,
    (arg_max(AMT_BALANCE, MONTHS_BALANCE)
        - arg_min(AMT_BALANCE, MONTHS_BALANCE))          AS cc_balance_trend
FROM cc
GROUP BY SK_ID_CURR;
