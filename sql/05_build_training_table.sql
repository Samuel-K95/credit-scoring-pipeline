-- Join application base with child feature tables into the modelling frame.
-- One row per SK_ID_CURR. Count-style columns coalesce to 0 (no history).
-- Depends on views: app; tables: installments_features, bureau_features,
-- previous_app_features, pos_features, cc_features.
CREATE OR REPLACE TABLE train_features AS
WITH app_base AS (
    SELECT
        SK_ID_CURR,
        TARGET,
        app_credit_to_income,
        app_annuity_to_income,
        app_credit_to_goods,
        AGE_YEARS        AS app_age_years,
        YEARS_EMPLOYED   AS app_years_employed,
        AMT_INCOME_TOTAL AS app_amt_income_total,
        AMT_CREDIT       AS app_amt_credit,
        AMT_ANNUITY      AS app_amt_annuity,
        EXT_SOURCE_1     AS app_ext_source_1,
        EXT_SOURCE_2     AS app_ext_source_2,
        EXT_SOURCE_3     AS app_ext_source_3,
        CNT_CHILDREN     AS app_cnt_children,
        CNT_FAM_MEMBERS  AS app_cnt_fam_members
    FROM app
)
SELECT
    a.*,
    COALESCE(i.inst_count, 0)          AS inst_count,
    i.inst_dpd_mean,
    i.inst_dpd_max,
    i.inst_late_rate,
    i.inst_underpay_share,
    i.inst_pay_ratio_mean,
    i.inst_late_rate_3m,
    i.inst_late_rate_6m,
    i.inst_late_rate_12m,
    COALESCE(b.bur_count, 0)           AS bur_count,
    COALESCE(b.bur_active_count, 0)    AS bur_active_count,
    COALESCE(b.bur_closed_count, 0)    AS bur_closed_count,
    b.bur_overdue_amt_sum,
    b.bur_max_overdue_amt,
    b.bur_max_days_overdue,
    b.bur_utilisation,
    b.bur_history_length_days,
    b.bur_amt_credit_sum_mean,
    b.bur_bb_dpd_share_mean,
    COALESCE(b.bur_bb_months_total, 0) AS bur_bb_months_total,
    COALESCE(p.prev_count, 0)          AS prev_count,
    p.prev_approved_rate,
    p.prev_refused_rate,
    p.prev_days_since_last,
    p.prev_amt_application_mean,
    p.prev_amt_credit_mean,
    p.prev_credit_to_application_mean,
    COALESCE(ps.pos_months_count, 0)      AS pos_months_count,
    ps.pos_dpd_mean,
    ps.pos_dpd_max,
    ps.pos_dpd_def_max,
    COALESCE(ps.pos_delinquency_count, 0) AS pos_delinquency_count,
    ps.pos_future_inst_trend,
    COALESCE(c.cc_months_count, 0)     AS cc_months_count,
    c.cc_utilisation_mean,
    c.cc_utilisation_max,
    c.cc_dpd_max,
    c.cc_drawings_mean,
    c.cc_balance_trend
FROM app_base a
LEFT JOIN installments_features i    USING (SK_ID_CURR)
LEFT JOIN bureau_features b          USING (SK_ID_CURR)
LEFT JOIN previous_app_features p    USING (SK_ID_CURR)
LEFT JOIN pos_features ps            USING (SK_ID_CURR)
LEFT JOIN cc_features c              USING (SK_ID_CURR);
