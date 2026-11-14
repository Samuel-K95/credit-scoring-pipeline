# Notebook 01 - Download and explore raw data

Source notebook: `notebooks/01_download_and_explore.ipynb`

## Purpose

Pull the Home Credit Default Risk tables from Kaggle into `data/raw/`, then
profile them without ever holding more than one table in memory at a time.
Only `application_train.csv` is small enough to load in full; the child tables
are scanned in chunks to get row counts, missingness, and join-key coverage.

## Cell by cell

### Cell 0 (markdown) - Title and prerequisites
Explains the goal, the memory-light approach, and the Kaggle setup needed
(accepted competition rules, `~/.kaggle/kaggle.json`, the `kaggle` CLI). Notes
that a 403 error means the competition rules were not accepted yet, and that a
kernel restart is needed if notebook ran out of RAM before.

### Cell 1 (code) - Imports and constants
Imports `gc`, `subprocess`, `zipfile`, `pathlib`, numpy, pandas. Resolves
`PROJECT_ROOT` (works whether the notebook runs from the repo root or from
`notebooks/`), adds it to `sys.path`, and imports `DAYS_EMPLOYED_ANOMALY`,
`RAW_DATA_DIR`, `RAW_TABLE_FILES` from `src.config`. Defines the competition
slug, the zip name/path, `CHUNKSIZE = 500000`, and `SAMPLE_ROWS = 50000`.

Output: a tuple confirming the project root, the raw data directory, the chunk
size (500000), and the sample size (50000).

### Cell 2 (markdown) - Download instructions
Gives the exact Kaggle CLI command and the seven CSVs that must exist after
unzipping. States that download is skipped when all expected CSVs are present.

### Cell 3 (code) - Download and unzip
`expected_csvs_present` checks for all required tables. `download_competition`
runs the Kaggle CLI. `unzip_archives` extracts any `.zip` into `data/raw/`. The
logic downloads only if neither the zips nor the CSVs exist, then unzips, then
reports any still-missing tables.

Output: `All expected CSVs already present. Skipping download.` and
`All expected tables present.`, followed by an empty missing list `[]`. On this
machine the data was already downloaded.

### Cell 4 (markdown) - "What is on disk?"
Section header for the file listing.

### Cell 5 (code) - File inventory
Walks `data/raw/`, skips `.gitkeep`, and records each file name, extension, size
in MB, and whether it is one of the expected tables, sorted largest first.

Output: a table. Largest files are `installments_payments.csv` (689.62 MB) and
the competition zip (688.19 MB), then `credit_card_balance.csv` (404.91),
`previous_application.csv` (386.21), `POS_CASH_balance.csv` (374.51),
`bureau_balance.csv` (358.19), `bureau.csv` (162.14), `application_train.csv`
(158.44). Also present but not expected: `application_test.csv`,
`sample_submission.csv`, and the column-description file.

### Cell 6 (markdown) - Memory-light summary plan
Describes the chunked profiling approach: header-only column names, exact row
and null counts via chunked reads, and a small sample for previews.

### Cell 7 (code) - Chunked table profiling
`scan_csv` reads each table in 500k-row chunks, accumulating row count and
per-column null counts without keeping the full frame. For each expected table
it prints the shape, shows the 12 columns with the highest missing percentage,
previews three sample rows, and finally builds a summary frame.

Output highlights:
- `application_train.csv`: 307511 rows, 122 cols. The most-missing columns are
  building-metadata fields like `COMMONAREA_*` (about 69.9 percent missing).
- `bureau.csv`: 1716428 rows, 17 cols; `AMT_ANNUITY` is 71.47 percent missing.
- `bureau_balance.csv`: 27299925 rows, 3 cols, no missing.
- `previous_application.csv`: 1670214 rows, 37 cols; `RATE_INTEREST_*` is
  99.64 percent missing.
- `installments_payments.csv`: 13605401 rows, 8 cols.
- `POS_CASH_balance.csv`: 10001358 rows, 8 cols.
- `credit_card_balance.csv`: 3840312 rows, 23 cols.
The final summary table lists each table's rows, columns, and the percentage of
columns that contain any missing values (for example 54.92 percent for
application, 0 percent for bureau_balance).

### Cell 8 (markdown) - Application target and keys
Notes that `application_train.csv` is small enough to load fully.

### Cell 9 (code) - Target and key checks
Loads application in full and inspects `TARGET`, the primary key, and the
`DAYS_EMPLOYED` anomaly.

Output: shape (307511, 122); `TARGET` is 282686 non-defaults and 24825 defaults,
a default rate of 0.0807 (about 8 percent). `SK_ID_CURR` is unique with zero
duplicates. The `DAYS_EMPLOYED == 365243` sentinel appears in 55374 rows
(18.01 percent), and the raw `DAYS_EMPLOYED` describe shows a max of 365243,
confirming the sentinel spike.

### Cell 10 (markdown) - Join-key coverage plan
Explains that only ID columns are read, in chunks, to measure how many
applications have child records.

### Cell 11 (code) - Join-key coverage
`unique_ids_from_csv` collects the unique IDs per child table in chunks. For
each child table it computes the overlap with application IDs and the percentage
of applications that have at least one child record. `bureau_balance` is checked
via `SK_ID_BUREAU`.

Output: coverage table. About 85.69 percent of applications have a bureau
record, 94.65 percent a previous application, 94.84 percent installments,
94.12 percent POS, and only 28.26 percent a credit card record. For bureau
balance, 45.11 percent of bureau IDs have monthly balance rows.

### Cell 12 (markdown) - Optional peek
Explains the optional single-table sample preview.

### Cell 13 (code) - Peek one table
Samples `installments_payments.csv` (50000 rows), prints its shape, dtypes,
head, and a transposed describe for the first 20 columns.

Output: sample shape (50000, 8), the dtype listing, three head rows, and summary
statistics (for example `DAYS_INSTALMENT` averages around -1047 days).

### Cell 14 (markdown) - Notes for next steps
Reminders: raw files are gitignored, the zip can be deleted after extract, the
next notebook does deeper data quality, positive relative-day offsets must not
be used as application-time features, and later stages should prefer DuckDB or
chunked IO.

## Key takeaways

- The dataset is large and wide, with one main application table and six child
  tables joined on `SK_ID_CURR` (and bureau balance on `SK_ID_BUREAU`).
- The target is imbalanced at about 8 percent default.
- Several fields carry the `365243` day sentinel and heavy missingness that the
  next notebooks address before feature engineering.
