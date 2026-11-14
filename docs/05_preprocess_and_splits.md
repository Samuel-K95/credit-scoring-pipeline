# Notebook 05 - Preprocessing and stratified splits

Source notebook: `notebooks/05_preprocess_and_splits.ipynb`

## Purpose

Create the reproducible data partitions used by every modelling notebook: a
stratified 80/20 holdout and stratified 5-fold cross-validation indices on the
train partition. Also demonstrate the train-only preprocessing pipeline
(missing indicators, median imputation, outlier capping) and prove it does not
leak, then save the split indices to disk.

Note: this notebook was committed without stored cell outputs, so the entries
below describe what each cell computes and the output it produces when run.

## Cell by cell

### Cell 0 (markdown) - Title and pipeline description
Describes the three preprocessing steps (missing indicators for high-signal
sparse fields, median imputation, winsorise/cap at the train 1st and 99th
percentiles), notes that all features are already numeric (categoricals were
aggregated away in Stage 3), and that split indices are saved for
reproducibility.

### Cell 1 (code) - Imports and config
Standard imports, resolves `PROJECT_ROOT`, imports config values (`HOLDOUT_SIZE`,
`N_CV_FOLDS`, `RANDOM_SEED`, path constants, `TARGET_COL`, `ID_COL`) and the
preprocessing helpers from `src.preprocess` (`HIGH_SIGNAL_MISSING`,
`build_preprocess_pipeline`, `feature_columns`, `make_splits`).
Output: prints the seed, holdout fraction, CV fold count, and the features path.

### Cell 2 (code) - Load features and view missingness
Reads `train_features.parquet`, selects feature columns (excluding id and
target), and plots a horizontal bar chart of percent-missing per feature.
Output: prints the matrix shape, the target rate (about 0.0807), and the feature
count, plus the missingness chart (the sparse building and external-score flags
are the most missing).

### Cell 3 (markdown) - Split section header; notes holdout is carved first and
CV folds are defined only on the train partition.

### Cell 4 (code) - Build and verify splits
Calls `make_splits(y)` to produce train and holdout indices plus the CV folds.
Prints the train and holdout sizes and their target rates (which match, since
the split is stratified) and each fold's sizes and validation default rate.
Asserts the train and holdout index sets are disjoint and together cover every
row, then prints `split integrity ok`.

### Cell 5 (markdown) - Preprocess section header; points to
`build_preprocess_pipeline` (missing flags then median impute then p01/p99 cap).

### Cell 6 (code) - Build the preprocess pipeline
Selects the high-signal missing-indicator columns that exist in the feature set
and builds the (unfitted) pipeline.
Output: prints the chosen missing-indicator columns and displays the sklearn
Pipeline object.

### Cell 7 (markdown) - Before/after section header.

### Cell 8 (code) - Fit on train only, check holdout
Fits the pipeline on the train partition, transforms both train and holdout, and
asserts zero remaining NaNs in each. Shows the capping effect on
`app_amt_income_total` (before vs after) and prints the learned train quantile
bounds for that column.
Output: the transformed shapes, zero NaN counts, the before/after histogram, and
the capping bounds. This cell demonstrates that imputation and capping use only
train statistics.

### Cell 9 (code) - Missing-indicator rates
Collects the `*_was_missing` indicator columns and plots their mean rate on
train, then displays the rates as a table. These confirm which sparse fields
carry the most missingness signal.

### Cell 10 (code) - Fold-wise leakage spot-check
For the first two CV folds, fits a fresh preprocess pipeline on the fold-train
rows and transforms the fold-validation rows, asserting no NaNs in either. This
proves that when preprocessing is refit per fold there is no leakage from
validation into the fitted statistics.
Output: `fold 0/1: transform ok` lines and `CV fold preprocess spot-check
passed`.

### Cell 11 (markdown) - Save section header.

### Cell 12 (code) - Save split indices
Writes `split_indices.json` with the seed, holdout size, fold count, row count,
the train and holdout index lists, the CV folds, the feature column list, the
high-signal missing columns, and the train/holdout target rates. Prints the
output path and the sizes.
Output: `wrote .../split_indices.json`, plus the train/holdout/fold counts.

### Cell 13 (markdown) - Summary
Recaps the stratified 80/20 holdout with matched target rates, the stratified
5-fold CV on train only, the preprocess order, the train-only fitting rule, and
the production entrypoint (`python -m src.preprocess` / `make preprocess`).

## Key takeaways

- The holdout is carved once and never touched during tuning; CV folds live only
  inside the train partition.
- Preprocessing is fit on train (or fold-train) only, and the notebook proves
  this with explicit no-NaN assertions on the held-out rows.
- The saved `split_indices.json` makes every later modelling run reproducible
  against the exact same partitions.
