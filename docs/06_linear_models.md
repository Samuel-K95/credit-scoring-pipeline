# Notebook 06 - Linear models (logistic regression)

Source notebook: `notebooks/06_linear_models.ipynb`

## Purpose

Build the interpretable baseline that every other model family must beat:
a logistic regression scorecard backbone. The notebook tunes regularisation and
class weighting with cross-validated grid search on a search sample, shows
bias/variance diagnostics, plots an epoch-wise train/validation loss curve, and
evaluates the selected model once on the untouched holdout. All preprocessing is
refit inside each CV fold through `src.modeling.make_pipeline`, so there is no
leakage.

## Cell by cell

### Cell 0 (markdown) - Title and plan
States the four steps: tune with GridSearchCV, show validation and learning
curves, plot the loss curve, and evaluate on holdout.

### Cell 1 (code) - Imports and data
Resolves the project root and imports the modelling toolkit from `src.modeling`
(metrics, grid search, pipeline, subsample, report helpers, plots). Loads the
modelling data via `load_modeling_data()`.
Output: the data summary - 246005 train rows, 61502 holdout, 52 features, train
and holdout target rate 0.0807, 5 folds, and a positive-class weight of 11.39
(the negative/positive ratio used for cost-sensitive learning).

### Cell 2 (code) - Search sample
Sets `SEARCH_FRAC = 0.2` and builds a stratified 20 percent subsample of the
train partition for fast tuning. The holdout is never touched.
Output: the sample summary - 49201 train rows, same 52 features and 0.0807 rate.

### Cell 3 (markdown) - Hyperparameter search header; grid over C, penalty, and
class weight, with AUC driving selection.

### Cell 4 (code) - Grid search
Runs `GridSearchCV` over a `LogisticRegression(solver="liblinear")` with
`C` in {0.01, 0.1, 1.0, 3.0}, `penalty` l2, and `class_weight` in
{None, balanced}, refit on ROC AUC.
Output: 8 candidates times 5 folds, best ROC AUC 0.7271 with C=1.0,
class_weight balanced, penalty l2. The results table shows balanced weighting
clearly beats unweighted (the top four rows are all balanced, around 0.71 to
0.73 AUC, versus about 0.58 to 0.59 unweighted), confirming the class weighting
matters on this imbalanced target.

### Cell 5 (markdown) - Refit header; explains the selected config is retrained
on all 246k training rows for honest holdout evaluation.

### Cell 6 (code) - Refit on full train
Strips the `model__` prefix from the best params and fits a fresh pipeline on
the full training partition.
Output: prints the selected params and displays the fitted Pipeline, which shows
the nested preprocess steps (missing flags on the high-signal columns, median
impute, quantile cap) and the final balanced logistic regression.

### Cell 7 (markdown) - Validation and learning curves header.

### Cell 8 (code) - Validation and learning curves
Plots a validation curve over `C` (semilog) and a learning curve over training
size, both on the search sample with the saved folds.
Output: two figures. The validation curve shows AUC versus regularisation
strength; the learning curve shows train versus validation AUC as data grows,
with the small gap indicating regularisation is controlling variance.

### Cell 9 (markdown) - Loss curve header; explains logistic regression has no
native iteration history, so an equivalent log-loss SGD model is fit
incrementally to record per-epoch loss.

### Cell 10 (code) - Loss curve
Calls `fit_logreg_sgd_curve` (60 epochs) and plots train versus validation log
loss per epoch.
Output: the loss-curve figure, showing both curves falling and flattening, which
is the linear-model analogue of a boosting loss curve.

### Cell 11 (markdown) - Holdout evaluation header.

### Cell 12 (code) - Train vs holdout metrics
Builds the train-versus-holdout metrics table for the final model.
Output: train and holdout are very close (ROC AUC 0.7115 vs 0.7128, KS 0.3142
vs 0.3148, and similar log loss and Brier), which shows the linear model is not
overfitting. The best-F1 threshold is 0.60.

### Cell 13 (code) - Holdout diagnostics
Plots ROC and PR curves, a calibration curve, a score distribution, and a
confusion matrix at the best-F1 threshold.
Output: two figure groups (the 2x2 diagnostic panel and the confusion matrix).

### Cell 14 (markdown) - Coefficients header; standardised-scale coefficients
ranked by absolute weight.

### Cell 15 (code) - Coefficients
Extracts the preprocessed feature names and the model coefficients and shows the
top 20 by absolute value.
Output: the strongest drivers are the external scores
(`app_ext_source_2` -0.47, `app_ext_source_3` -0.28, both protective),
`app_credit_to_goods` (+0.23, risk-increasing), `bur_utilisation` (+0.23), and
several `*_was_missing` indicators, confirming that both the values and their
missingness carry signal.

### Cell 16 (markdown) - Takeaways
Logistic regression gives the interpretable baseline, balanced weighting
compensates for the 8 percent default rate, and the curves show variance is
controlled. Later families are judged against this.

## Key takeaways

- Baseline holdout ROC AUC about 0.713, KS about 0.315.
- Tuning on a 20 percent sample then refitting on the full train keeps search
  fast while the reported model is trained on all the data.
- Coefficients are sensible and interpretable, which is why this model anchors
  the scorecard narrative.
