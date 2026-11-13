-- Previous application aggregates to one row per SK_ID_CURR.
-- Prefix: prev_
-- Leakage guard: only decisions at/before application (DAYS_DECISION <= 0).
-- Depends on view: previous_application.
CREATE OR REPLACE TABLE previous_app_features AS
SELECT
    SK_ID_CURR,
    COUNT(*)                                                          AS prev_count,
    AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END) AS prev_approved_rate,
    AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Refused'  THEN 1 ELSE 0 END) AS prev_refused_rate,
    (-MAX(DAYS_DECISION))                                             AS prev_days_since_last,
    AVG(AMT_APPLICATION)                                              AS prev_amt_application_mean,
    AVG(AMT_CREDIT)                                                   AS prev_amt_credit_mean,
    AVG(AMT_CREDIT / NULLIF(AMT_APPLICATION, 0))                      AS prev_credit_to_application_mean
FROM previous_application
WHERE DAYS_DECISION <= 0
GROUP BY SK_ID_CURR;
