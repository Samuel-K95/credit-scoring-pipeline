-- Bureau and bureau_balance aggregates to one row per SK_ID_CURR.
-- Prefix: bur_ (bureau_balance rollup prefixed bur_bb_).
-- Leakage guard: only records with DAYS_CREDIT <= 0 and MONTHS_BALANCE <= 0.
-- Depends on views: bureau, bureau_balance.
CREATE OR REPLACE TABLE bureau_features AS
WITH bb AS (
    SELECT
        SK_ID_BUREAU,
        COUNT(*) AS months,
        AVG(CASE WHEN STATUS IN ('1', '2', '3', '4', '5') THEN 1 ELSE 0 END) AS dpd_share
    FROM bureau_balance
    WHERE MONTHS_BALANCE <= 0
    GROUP BY SK_ID_BUREAU
),
bb_curr AS (
    SELECT
        b.SK_ID_CURR,
        AVG(bb.dpd_share) AS bur_bb_dpd_share_mean,
        SUM(bb.months)    AS bur_bb_months_total
    FROM bureau b
    JOIN bb USING (SK_ID_BUREAU)
    WHERE b.DAYS_CREDIT <= 0
    GROUP BY b.SK_ID_CURR
),
bur AS (
    SELECT
        SK_ID_CURR,
        COUNT(*)                                                   AS bur_count,
        SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END)  AS bur_active_count,
        SUM(CASE WHEN CREDIT_ACTIVE = 'Closed' THEN 1 ELSE 0 END)  AS bur_closed_count,
        SUM(AMT_CREDIT_SUM_OVERDUE)                                AS bur_overdue_amt_sum,
        MAX(AMT_CREDIT_MAX_OVERDUE)                                AS bur_max_overdue_amt,
        MAX(CREDIT_DAY_OVERDUE)                                    AS bur_max_days_overdue,
        SUM(AMT_CREDIT_SUM_DEBT) / NULLIF(SUM(AMT_CREDIT_SUM), 0)  AS bur_utilisation,
        (-MIN(DAYS_CREDIT))                                       AS bur_history_length_days,
        AVG(AMT_CREDIT_SUM)                                        AS bur_amt_credit_sum_mean
    FROM bureau
    WHERE DAYS_CREDIT <= 0
    GROUP BY SK_ID_CURR
)
SELECT
    bur.*,
    bb_curr.bur_bb_dpd_share_mean,
    bb_curr.bur_bb_months_total
FROM bur
LEFT JOIN bb_curr USING (SK_ID_CURR);
