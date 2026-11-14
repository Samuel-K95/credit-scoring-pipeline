# Notebook 04 - Feature engineering (DuckDB SQL)

Source notebook: `notebooks/04_feature_engineering.ipynb`

## Purpose

Aggregate each child table down to one row per `SK_ID_CURR` using DuckDB SQL,
then left-join everything onto the cleaned application table to build the single
modelling table `data/processed/train_features.parquet`. Every child
aggregation filters to relative offsets available at application time (day or
month offset `<= 0`), and the notebook asserts this explicitly as a leakage
guard. Feature names are prefixed by source: `inst_`, `bur_`, `prev_`, `pos_`,
`cc_`, `app_`.

Note: this notebook was committed without stored cell outputs, so the entries
below describe what each cell computes and the output it produces when run.

## Cell by cell

### Cell 0 (markdown) - Title, sources, leakage rule
Lists the inputs (the clean parquets from Stage 2 plus the raw CSVs for the
large tables cleaned inline in SQL), the column prefixes, and the leakage rule.

### Cell 1 (code) - Imports and DuckDB connection
Imports duckdb, matplotlib, pandas, seaborn. Resolves `PROJECT_ROOT`, imports
the config paths, converts each source path to a DuckDB-friendly posix string,
opens a DuckDB connection, and sets `PRAGMA threads=4`.
Output: prints the DuckDB version and `sources ready`.

### Cell 2 (markdown) - Leakage guard helper
Explains that each child aggregation filters to application-time offsets and
that the filtered sources are verified to contain no positive offsets.

### Cell 3 (code) - Leakage assertions
`check_max_offset` runs a min/max query on a filtered source and asserts the max
offset is `<= 0`. It is called for bureau `DAYS_CREDIT`, bureau_balance
`MONTHS_BALANCE`, previous application `DAYS_DECISION`, installments
`DAYS_INSTALMENT`, POS `MONTHS_BALANCE`, and credit card `MONTHS_BALANCE`.
Output: prints each offset range and finally `No positive offsets in any
filtered source.` If any positive offset slipped through, the assertion would
stop the notebook.

### Cell 4 (markdown) - inst_ section header.

### Cell 5 (code) - Installment features
SQL that, filtering to `DAYS_INSTALMENT <= 0`, derives per-payment days past
due, payment ratio, underpayment and late flags, then aggregates per applicant
into: `inst_count`, `inst_dpd_mean`, `inst_dpd_max`, `inst_late_rate`,
`inst_underpay_share`, `inst_pay_ratio_mean`, and late-rate windows over the
last 3, 6, and 12 months (`inst_late_rate_3m/6m/12m`). Prints the shape, head,
and summary, plus histograms of the late rate and payment ratio.

### Cell 6 (markdown) - bur_ section header.

### Cell 7 (code) - Bureau features
SQL over `bureau_clean.parquet` filtered to `DAYS_CREDIT <= 0`, producing
counts (`bur_count`, active, closed), overdue amounts, max days overdue,
utilisation (`SUM(debt) / SUM(credit_sum)`), history length (`-MIN(DAYS_CREDIT)`),
and mean credit sum. Prints shape, head, and summary.

### Cell 8 (markdown) - bur_bb_ section header.

### Cell 9 (code) - Bureau balance features
SQL that aggregates monthly bureau balance (DPD share and month count per bureau
credit) and joins through the bureau map to the applicant, producing
`bur_bb_dpd_share_mean` and `bur_bb_months_total`. Prints shape, head, summary.

### Cell 10 (markdown) - prev_ section header.

### Cell 11 (code) - Previous application features
SQL over `previous_application_clean.parquet` filtered to `DAYS_DECISION <= 0`:
`prev_count`, approval and refusal rates, time since last decision
(`-MAX(DAYS_DECISION)`), mean requested and approved amounts, and the
mean credit-to-application ratio. Prints shape, head, summary, and a histogram
of approval rate.

### Cell 12 (markdown) - pos_ section header.

### Cell 13 (code) - POS cash features
SQL over POS cash filtered to `MONTHS_BALANCE <= 0`: month count, DPD mean/max,
max DPD-def, delinquency count, and a future-instalment trend computed with
`arg_max - arg_min` over the month axis (`pos_future_inst_trend`). Prints shape,
head, summary.

### Cell 14 (markdown) - cc_ section header.

### Cell 15 (code) - Credit card features
SQL over credit card filtered to `MONTHS_BALANCE <= 0`: month count, mean and
max utilisation (with a safe divide against a zero limit), max DPD, mean
drawings, and a balance trend via `arg_max - arg_min`. Prints shape, head,
summary.

### Cell 16 (markdown) - app_ section header.

### Cell 17 (code) - Application base features
SQL over `application_train_clean.parquet` selecting the key, `TARGET`, the
three derived ratios from Stage 2, age and employment years, raw amounts, the
three external scores, and household counts, each renamed with the `app_`
prefix. Prints shape and the target rate.

### Cell 18 (markdown) - Join section header; explains that left joins keep
applicants without child history as NaN, and that count-style columns are then
filled with 0 because "no history" is itself informative.

### Cell 19 (code) - Join into the modelling table
Registers each aggregate frame with DuckDB and left-joins them all onto the
application base by `SK_ID_CURR`. Then fills the count-like columns (those
ending in `_count`, `_months_total`, `_months_count`, `_delinquency_count`) with
0. Prints the joined shape and the list of count columns filled.

### Cell 20 (code) - Integrity checks
Asserts the key is unique and non-null and that the row count equals the
application base (the join did not duplicate rows). Counts the prefixed feature
columns and plots feature missingness, where gaps are left-join misses that mean
"no history".

### Cell 21 (code) - Univariate signal check
Computes the Pearson correlation of each numeric feature with `TARGET` and plots
the top positive and negative. This is a sanity check that the strongest
correlates (the external scores, late rates, and so on) point in sensible
directions.

### Cell 22 (code) - Write the feature table
Writes `data/processed/train_features.parquet` and closes the DuckDB
connection. Prints the output path and final shape.

### Cell 23 (markdown) - Summary
One row per applicant with the prefixed features, the asserted leakage guard,
and the output path; points to porting the SQL into `sql/*.sql` and
`src/build_features.py` plus a leakage unit test.

## Key takeaways

- All history features are aggregated with DuckDB SQL under a strict
  application-time filter, and the filter is asserted, not just assumed.
- Left joins plus zero-filled counts encode "no prior history" as a real signal
  rather than as missing data.
- The result is a single numeric modelling table that every later stage loads.
