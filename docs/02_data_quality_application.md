# Notebook 02 - Data quality: application_train

Source notebook: `notebooks/02_data_quality_application.ipynb`

## Purpose

Inspect every application-level feature one at a time, visualize its
distribution, clean dirty values, and record a decision for each. The output is
a cleaned parquet plus a JSON decision log. The guiding rule is that any
imputation that needs statistics (medians, etc.) is deferred to the train-only
preprocessing stage, so here missing values are kept as NaN with explicit
`*_MISSING` indicator columns rather than being filled.

The notebook has 34 code cells. Many are a repeating pattern applied per
feature: print a report, plot a distribution, apply a decision, and call
`record(...)`. The entries below describe every cell; the repetitive feature
cells are summarized with their key numbers and the decision taken.

## Setup cells

### Cell 0 (markdown) - Title and scope
Feature-by-feature quality for `application_train.csv` only; child tables come
in notebook 03.

### Cell 1 (code) - Imports and config
Standard imports (json, pathlib, matplotlib, numpy, pandas, seaborn), resolves
`PROJECT_ROOT`, imports `DAYS_EMPLOYED_ANOMALY`, `PROCESSED_DATA_DIR`,
`RAW_DATA_DIR`. Creates the processed directory, starts an empty `DECISIONS`
list, and sets `RANDOM_STATE = 42`.
Output: prints the project root and the anomaly code 365243.

### Cell 2 (code) - Helper functions
Defines the reusable helpers used throughout:
- `record(feature, issue, action, detail)` appends a decision and prints it.
- `plot_numeric`, `plot_numeric_by_target`, `plot_categorical` for charts.
- `before_after_numeric` for side-by-side before/after histograms.
- `feature_report` prints dtype, missing count and percent, unique count, and a
  describe or value-counts block.
No output (definitions only).

### Cell 3 (markdown) - Load section header.

### Cell 4 (code) - Load application_train
Reads the CSV in full into `df`, makes a working copy `clean`, and plots the
missingness bar chart for all columns that have any missing values.
Output: shape (307511, 122), memory about 536.7 MB, and that 67 of 122 columns
have missing values.

## Per-feature cells

### Cell 5 / 6 - TARGET
Report plus class-balance bar. Confirms values are in {0, 1} with no missing and
a mean of 0.0807. Decision: keep as-is.

### Cell 7 / 8 - SK_ID_CURR
Primary key, 307511 unique, zero duplicates. Decision: keep as-is.

### Cell 9 / 10 - CODE_GENDER
Values F (202448), M (105059), and the dirty `XNA` (4 rows). Decision: drop the
4 XNA rows (do not invent a gender). After drop the category is clean.

### Cell 11 / 12 - NAME_CONTRACT_TYPE
Cash loans (278232) vs Revolving loans (29275), no missing. Decision: keep.

### Cell 13 / 14 - FLAG_OWN_CAR
Y/N flag, no missing. Decision: keep.

### Cell 15 / 16 - FLAG_OWN_REALTY
Y/N flag, no missing. Decision: keep.

### Cell 17 / 18 - CNT_CHILDREN
Integer count, max 19, heavy right tail. Decision: keep raw, optionally cap in
preprocessing later.

### Cell 19 / 20 - AMT_INCOME_TOTAL
Heavily right-skewed income (max 117,000,000). Adds a log1p helper column for
EDA and shows before/after plus a by-target view. Decision: keep raw for
modelling, use log1p only for EDA, cap on train only in preprocessing.

### Cell 21 / 22 - AMT_CREDIT
Right-skewed, no missing, all positive. Adds a log1p EDA view. Decision: keep
raw, log1p for EDA.

### Cell 23 / 24 - AMT_ANNUITY
12 missing values. Adds `AMT_ANNUITY_MISSING` flag (rate about 0.00004).
Decision: keep NaN plus flag, impute in train-only preprocessing.

### Cell 25 / 26 - AMT_GOODS_PRICE
278 missing (0.09 percent). Shows the tight linear relationship between goods
price and credit amount. Adds a missing flag. Decision: keep NaN plus flag.

### Cell 27 / 28 - NAME_TYPE_SUITE
1292 missing categorical. Decision: fill NaN with an explicit `Missing`
category.

### Cell 29 / 30 - NAME_INCOME_TYPE
8 categories, no missing (Working dominates). Decision: keep.

### Cell 31 / 32 - NAME_EDUCATION_TYPE
5 categories, no missing. Decision: keep.

### Cell 33 / 34 - NAME_FAMILY_STATUS
Contains 2 `Unknown` rows. Decision: map Unknown to `Missing`.

### Cell 35 / 36 - NAME_HOUSING_TYPE
6 categories, no missing. Decision: keep.

### Cell 37 / 38 - REGION_POPULATION_RELATIVE
Continuous, no missing. Decision: keep.

### Cell 39 / 40 - DAYS_BIRTH
Negative day offset. Derives `AGE_YEARS = -DAYS_BIRTH / 365.25` (range about
20.5 to 69.1 years), keeps the raw column too. No positive values. Decision:
derive age, keep raw.

### Cell 41 / 42 - DAYS_EMPLOYED (known dirty)
The `365243` sentinel appears in 55374 rows (18.01 percent). Sets the sentinel
to NaN, adds `DAYS_EMPLOYED_ANOMALY_FLAG`, and derives `YEARS_EMPLOYED`. Shows
before/after plots and a by-target view. Decision: null the sentinel, add the
flag, derive years.

### Cell 43 / 44 - DAYS_REGISTRATION
Negative offset, no positive values. Derives `YEARS_REGISTRATION`. Decision:
derive years, keep raw.

### Cell 45 / 46 - DAYS_ID_PUBLISH
Negative offset. Derives `YEARS_ID_PUBLISH`. Decision: derive years.

### Cell 47 / 48 - OWN_CAR_AGE
65.99 percent missing, but the crosstab shows the missingness is structural:
essentially all missing rows are `FLAG_OWN_CAR == N` (99.9975 percent).
Decision: keep NaN plus `OWN_CAR_AGE_MISSING`, do not impute an age for people
with no car.

### Cell 49 / 50 - OCCUPATION_TYPE
31.35 percent missing categorical. Decision: fill NaN with `Missing`.

### Cell 51 / 52 - CNT_FAM_MEMBERS
2 missing; the consistency check finds zero rows where family members are fewer
than children. Adds a missing flag. Decision: keep NaN plus flag, do not
force-fix.

### Cell 53 / 54 - ORGANIZATION_TYPE
`XNA` appears 55374 times and overlaps exactly with the employment anomaly
(55374), confirming XNA here means unemployed. Decision: map XNA to `Missing`.

### Cell 55 / 56 - EXT_SOURCE_1/2/3
Loops over the three external credit scores. Missingness: EXT_SOURCE_1 is
56.38 percent, EXT_SOURCE_2 is 0.21 percent, EXT_SOURCE_3 is 19.83 percent.
Each gets a `*_MISSING` flag. The by-target plots show these scores separate
defaulters from non-defaulters well, which foreshadows their later dominance in
the model. Decision: keep NaN plus flags, impute train-only later.

### Cell 57 / 58 - AMT_REQ_CREDIT_BUREAU_*
Loops over the six bureau enquiry-count columns (HOUR, DAY, WEEK, MON, QRT,
YEAR). All share the same 13.50 percent missingness (they go missing together).
Each gets a missing flag. Decision: keep NaN plus flags.

### Cell 59 / 60 - Housing / building metadata
Collects the 47 sparse `*_AVG`, `*_MODE`, `*_MEDI` building columns plus a few
named mode columns. For each it prints missingness (roughly 48 to 70 percent),
plots the distribution and the missing flag, then: numeric columns keep NaN plus
a `*_MISSING` flag; categorical columns are filled with `Missing`. This single
cell emits many decisions, one per building column.

### Cell 61 / 62 - Binary FLAG_/REG_/LIVE_ features
Audits the original binary flags (phone, document, region-mismatch flags). Each
is confirmed to be in {0, 1} with no missing and recorded as keep as-is. This
produces many keep decisions in one pass.

### Cell 63 / 64 - Derived application ratios
Builds three modelling ratios: `app_credit_to_income`, `app_annuity_to_income`,
`app_credit_to_goods`, each with a safe divide (replacing a zero denominator
with NaN). Shows distributions clipped at the 99th percentile and by-target
views. Decision: keep the ratios, watch for extreme values from tiny incomes.
The describe outputs show, for example, credit-to-income averaging about 3.96
and credit-to-goods averaging about 1.12.

## Save and summary

### Cell 65 (markdown) - Save header.

### Cell 66 (code) - Save cleaned table and decision log
Writes `data/processed/application_train_clean.parquet` and
`application_cleaning_decisions.json`. Prints the before/after row count
(307511 to 307507, the 4 XNA rows dropped), the cleaned column count (188, up
from 122 because of derived and flag columns), and 116 recorded decisions.
Displays the tail of the decision table and the counts of each action type
(41 keep as-is, 5 fill Missing, and so on).

### Cell 67 (markdown) - Summary
Lists the seven key cleaning actions and the two output artifacts, and points to
notebook 03 for the child tables.

## Key takeaways

- Every feature has an explicit, logged decision, giving a reproducible and
  auditable cleaning trail.
- The design choice is to flag and defer, not impute: missing values stay NaN
  with `*_MISSING` indicators so no train statistics leak at this stage.
- Known Home Credit quirks are handled: the `365243` employment sentinel, `XNA`
  placeholders in gender and organization, and structural missingness in
  car age.
