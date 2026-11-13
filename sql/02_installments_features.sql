-- Installments repayment behaviour aggregates to one row per SK_ID_CURR.
-- Prefix: inst_
-- Leakage guard: only instalments due at/before application (DAYS_INSTALMENT <= 0).
-- Depends on view: installments.
CREATE OR REPLACE TABLE installments_features AS
WITH inst AS (
    SELECT
        SK_ID_CURR,
        DAYS_INSTALMENT,
        (DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)             AS days_past_due,
        (AMT_PAYMENT / NULLIF(AMT_INSTALMENT, 0))          AS pay_ratio,
        CASE WHEN AMT_PAYMENT < AMT_INSTALMENT THEN 1 ELSE 0 END      AS is_underpay,
        CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1 ELSE 0 END AS is_late
    FROM installments
    WHERE DAYS_INSTALMENT <= 0
)
SELECT
    SK_ID_CURR,
    COUNT(*)                                                 AS inst_count,
    AVG(days_past_due)                                       AS inst_dpd_mean,
    MAX(days_past_due)                                       AS inst_dpd_max,
    AVG(is_late)                                             AS inst_late_rate,
    AVG(is_underpay)                                         AS inst_underpay_share,
    AVG(pay_ratio)                                           AS inst_pay_ratio_mean,
    AVG(CASE WHEN DAYS_INSTALMENT >= -90  THEN is_late END)  AS inst_late_rate_3m,
    AVG(CASE WHEN DAYS_INSTALMENT >= -180 THEN is_late END)  AS inst_late_rate_6m,
    AVG(CASE WHEN DAYS_INSTALMENT >= -365 THEN is_late END)  AS inst_late_rate_12m
FROM inst
GROUP BY SK_ID_CURR;
