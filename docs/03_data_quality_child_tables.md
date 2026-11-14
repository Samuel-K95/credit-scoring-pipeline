# Notebook 03 - Data quality: child tables

Source notebook: `notebooks/03_data_quality_child_tables.ipynb`

## Purpose

Apply the same inspect-visualize-clean-record discipline from notebook 02 to the
six child tables: bureau, bureau balance, previous application, installments,
POS cash, and credit card. Two tables (bureau and previous application) are
small enough to clean and write to parquet. The four very large tables are only
profiled here (on samples and chunked counts); their actual cleaning rules are
applied later inside the DuckDB SQL in notebook 04 to avoid rewriting multi-GB
files. The notebook has 30 code cells.

## Setup cells

### Cell 0 (markdown) - Title and scope
Describes the plan, the one-table-at-a-time memory discipline, and the leakage
reminder that offsets after application time must not feed application-time
features.

### Cell 1 (code) - Imports and config
Standard imports plus `gc`. Resolves `PROJECT_ROOT`, imports config paths and
`DAYS_EMPLOYED_ANOMALY` (reused as the cross-table day sentinel 365243). Sets
`CHUNKSIZE = 500000`, `SAMPLE_N = 80000`.
Output: prints project root and `day sentinel: 365243`.

### Cell 2 (code) - Helper functions
Defines `record` (now with a `table` field), the plotting helpers,
`feature_report`, `replace_day_sentinel` (returns a cleaned series plus an
anomaly flag), `sample_csv` (reads first chunks up to a sample size),
`chunked_value_counts`, and `chunked_null_and_sentinel` (full-data null and
sentinel counts without loading the whole table). No output.

## A. bureau.csv

### Cell 3 (markdown) - Section header for bureau.

### Cell 4 (code) - Load bureau
Loads the full table and plots missingness.
Output: shape (1716428, 17), memory about 512.1 MB.

### Cell 5 / 6 - Keys
Asserts no missing keys, zero duplicate `SK_ID_BUREAU`, and 305811 unique
`SK_ID_CURR`. Decisions: keep both keys.

### Cell 7 / 8 - CREDIT_ACTIVE
Categories Closed (1079273), Active (630607), Sold (6527), Bad debt (21). No
missing. Decision: keep.

### Cell 9 / 10 - CREDIT_CURRENCY
Almost entirely currency 1. Decision: keep but low value, may drop later.

### Cell 11 / 12 - DAYS_CREDIT
Negative offset, max 0, zero positive values. Decision: keep, require <= 0 for
application-time features.

### Cell 13 / 14 - CREDIT_DAY_OVERDUE
Mostly zero with a sparse positive tail (max 2792). Decision: keep as-is.

### Cell 15 / 16 - DAYS_CREDIT_ENDDATE
6.15 percent missing. Importantly, 602603 values are positive (planned end dates
after application) and 1007389 are negative. Adds a missing flag. Decision: keep
NaN plus flag; for PD features prefer closed or past information only (the
positive, forward-looking values are a leakage risk).

### Cell 17 / 18 - DAYS_ENDDATE_FACT
36.92 percent missing (missing when the credit is still active). Adds a missing
flag. Decision: keep NaN plus flag.

### Cell 19 / 20 - AMT_CREDIT_MAX_OVERDUE
65.51 percent missing, extreme right tail. Plots on a log1p scale. Adds a
missing flag. Decision: keep NaN plus flag.

### Cell 21 / 22 - AMT_CREDIT_SUM family
Loops over `AMT_CREDIT_SUM` (13 missing), `AMT_CREDIT_SUM_DEBT`
(15.01 percent), `AMT_CREDIT_SUM_LIMIT` (34.48 percent), and
`AMT_CREDIT_SUM_OVERDUE` (none). Columns with any missing get a `*_MISSING`
flag; the fully populated one is kept as-is. Note that debt and limit can be
negative in the raw data.

### Cell 23 / 24 - CNT_CREDIT_PROLONG
Mostly zero (max 9), no missing. Decision: keep.

### Cell 25 / 26 - CREDIT_TYPE
15 categories dominated by Consumer credit and Credit card, no missing.
Decision: keep.

### Cell 27 / 28 - DAYS_CREDIT_UPDATE
Has 17 positive values (dirty, since this should be <= 0). Sets positives to
NaN, adds `DAYS_CREDIT_UPDATE_POS_FLAG`, shows before/after. Decision: null the
positives plus flag.

### Cell 29 / 30 - AMT_ANNUITY (bureau)
71.47 percent missing. Adds a missing flag. Decision: keep NaN plus flag, use
cautiously in aggregates.

### Cell 31 (code) - Save bureau
Writes `bureau_clean.parquet` (shape 1716428 x 25, the extra columns being the
missing/anomaly flags) and frees memory.

## B. bureau_balance.csv

### Cell 32 (markdown) - Section header.

### Cell 33 (code) - Profile bureau_balance (large)
Uses chunked value counts for `STATUS` over the full 358 MB file and a sample
for `MONTHS_BALANCE`. STATUS distribution: C (closed) 13646993, 0 (no DPD)
7499507, X (unknown) 5810482, then small counts of 1 to 5 month DPD buckets.
Sample `MONTHS_BALANCE` is all <= 0 with no positives. Decisions: keep STATUS
(documenting the C/X/0-5 meaning), keep MONTHS_BALANCE but use only <= 0 at
feature time, and apply cleaning rules in SQL rather than rewriting the table.

## C. previous_application.csv

### Cell 34 (markdown) - Section header.

### Cell 35 (code) - Load previous_application
Loads the full table and plots missingness.
Output: shape (1670214, 37).

### Cell 36 / 37 - NAME_CONTRACT_STATUS
Approved (1036781), Canceled (316319), Refused (290678), Unused offer (26436),
no missing. Decision: keep (core for approval and refusal rates).

### Cell 38 / 39 - DAYS_DECISION
Negative offset, no positives and no sentinel. Decision: keep, filter to <= 0.

### Cell 40 / 41 - Day sentinel columns
Loops over `DAYS_FIRST_DRAWING`, `DAYS_FIRST_DUE`, `DAYS_LAST_DUE_1ST_VERSION`,
`DAYS_LAST_DUE`, `DAYS_TERMINATION`. Each is 40.30 percent NaN and carries the
365243 sentinel at varying rates (for example DAYS_FIRST_DRAWING is 55.95
percent sentinel, DAYS_TERMINATION 13.53 percent). Each gets the sentinel set to
NaN plus a `*_ANOMALY_FLAG`, with before/after plots. Decisions: null sentinel
plus flag for all five.

### Cell 42 / 43 - Amount columns
Loops over `AMT_APPLICATION` (no missing), `AMT_CREDIT` (1 missing),
`AMT_ANNUITY` (22.29 percent), `AMT_DOWN_PAYMENT` (53.64 percent),
`AMT_GOODS_PRICE` (23.08 percent). Plots on log1p scale. Missing columns get a
`*_MISSING` flag; fully populated ones are kept.

### Cell 44 / 45 - Categoricals with XNA/XAP
Loops over 12 categorical columns (contract type, cash loan purpose, reject
reason, client type, goods category, portfolio, product type, channel, seller
industry, yield group, product combination, payment type, type suite). For each
it maps the `XNA` and `XAP` placeholders and NaN to a single `Missing` category.
Some are heavily affected (for example NAME_CASH_LOAN_PURPOSE has about 1.6M
XNA/XAP rows; NAME_TYPE_SUITE has 820405 NaN). Decisions: map to Missing for
each.

### Cell 46 / 47 - Extra numeric columns
Loops over `RATE_DOWN_PAYMENT` (53.64 percent missing), `RATE_INTEREST_PRIMARY`
and `RATE_INTEREST_PRIVILEGED` (both 99.64 percent missing),  `CNT_PAYMENT`
(22.29 percent), `SELLERPLACE_AREA` (uses -1 as unknown). Missing columns get a
flag; `SELLERPLACE_AREA == -1` (762675 rows) is set to NaN plus a missing flag.

### Cell 48 (code) - Save previous_application
Writes `previous_application_clean.parquet` (shape 1670214 x 51).

## D. installments_payments.csv (large)

### Cell 49 (markdown) - Section header.

### Cell 50 (code) - Profile installments (large)
Uses chunked null/sentinel counts over the full 689 MB file: 13605401 rows, with
`DAYS_ENTRY_PAYMENT` and `AMT_PAYMENT` each having 2905 nulls and no sentinel.
On an 80k sample it derives `DAYS_LATE = entry - instalment` and
`PAYMENT_RATIO = payment / instalment`, confirming no positive day offsets.
Decisions: keep NaN on entry payment and derive late/underpay features only when
both day fields are present; derive the payment ratio and late flags in the
feature SQL; apply cleaning in SQL rather than rewriting the table.

## E. POS_CASH_balance.csv (large)

### Cell 51 (markdown) - Section header.

### Cell 52 (code) - Profile POS cash (large)
Chunked counts of `NAME_CONTRACT_STATUS` plus an 80k sample for
`MONTHS_BALANCE`, `CNT_INSTALMENT`, `CNT_INSTALMENT_FUTURE`, `SK_DPD`,
`SK_DPD_DEF`. Months balance is all <= 0; the two instalment-count columns have
about 26k nulls over the full data; DPD fields are mostly zero with long tails.
Decisions: use months <= 0, keep and aggregate DPD in SQL, handle instalment
counts carefully in SQL, no full rewrite.

## F. credit_card_balance.csv (large)

### Cell 53 (markdown) - Section header; notes the dataset typo `AMT_RECIVABLE`
is kept for join compatibility.

### Cell 54 (code) - Profile credit card (large)
80k sample for distributions (balance, limit, drawings, payments, DPD) plus a
utilisation proxy `AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL` and chunked status
counts. Balances can be negative; limits can be zero (so utilisation needs a
safe divide). Decisions: keep the typo column name, keep NaN and aggregate in
SQL, use months <= 0, safe-divide utilisation, no full rewrite.

## Save and summary

### Cell 55 (markdown) - Save header.

### Cell 56 (code) - Save decision log
Writes `child_tables_cleaning_decisions.json` recording the day sentinel, which
tables were written as clean parquets, which are SQL-only, and all 64 decisions.
Displays the per-table decision counts (bureau 17, previous_application 31,
and so on) and the tail of the decision table.

### Cell 57 (markdown) - Summary
A table summarizing how each child table is handled and the three output
artifacts, pointing to the Stage 2 porting and the switch to a premium model for
Stage 3 SQL features.

## Key takeaways

- Small tables (bureau, previous application) are cleaned and persisted; large
  tables are only profiled, with their cleaning rules deferred to SQL.
- The `365243` day sentinel and the `XNA`/`XAP`/`-1` placeholders are the
  recurring data-quality issues across child tables.
- Forward-looking fields such as `DAYS_CREDIT_ENDDATE` are flagged as leakage
  risks and restricted to past information for application-time features.
