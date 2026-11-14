# Notebook 08 - Ensemble models (bagging)

Source notebook: `notebooks/08_ensemble_models.ipynb`

## Purpose

Reduce the variance of a single tree by bagging many de-correlated trees. The
notebook tunes a random forest, plots a loss curve versus the number of trees,
compares the forest to extremely randomised trees, and evaluates on the holdout.
Because forests parallelise internally (`n_jobs=-1`), the grid search runs with
`n_jobs=1` to avoid CPU oversubscription.

## Cell by cell

### Cell 0 (markdown) - Title and plan
Lists the steps: tune the forest, show the trees loss curve, compare to extra
trees, and evaluate with importance.

### Cell 1 (code) - Imports and data
Same toolkit imports and `load_modeling_data()`.
Output: data summary (246005 train, 61502 holdout, 52 features, 0.0807, 5 folds,
pos_weight 11.39).

### Cell 2 (code) - Search sample
`SEARCH_FRAC = 0.3` builds a 30 percent subsample (73802 train rows).

### Cell 3 (markdown) - Random forest search header.

### Cell 4 (code) - Grid search
`GridSearchCV` over `RandomForestClassifier` with `n_estimators` in {300, 500},
`max_depth` in {8, 12, None}, `max_features` in {sqrt, 0.5}, `min_samples_leaf`
in {20, 50}, and `class_weight` in {balanced, balanced_subsample}; 48 candidates
times 5 folds with `n_jobs=1`.
Output: best ROC AUC 0.7581 with balanced_subsample, max_depth None,
max_features sqrt, min_samples_leaf 20, n_estimators 500. This was the slowest
search in the project (about 19292 seconds). The results table shows very high
train AUC (around 0.97) against a validation AUC around 0.758, the expected
bagging pattern where individual trees overfit but their average generalizes.

### Cell 5 (markdown) - Trees loss curve header; explains growing a warm-start
forest and recording loss as trees are added, where validation loss falls then
plateaus.

### Cell 6 (code) - Loss curve vs number of trees
Uses `warm_start_loss_curve` over {25, 50, 100, 150, 200, 300, 400} trees.
Output: the loss-versus-trees figure, where validation loss drops quickly then
flattens, showing more trees cut variance without overfitting.

### Cell 7 (markdown) - CV curves header.

### Cell 8 (code) - Validation and learning curves
Validation curve over `n_estimators` and a learning curve (both `n_jobs=1`).
Output: two figures.

### Cell 9 (markdown) - Forest vs extra trees header.

### Cell 10 (code) - Compare random forest and extra trees
Fits an `ExtraTreesClassifier` on the sample and compares both on the holdout.
Output: the two are close - random forest ROC AUC 0.7600 versus extra trees
0.7565. Note the random forest log loss (0.392) and Brier (0.116) are much lower
than extra trees (0.593 / 0.202), meaning the forest is better calibrated out of
the box.

### Cell 11 (markdown) - Holdout evaluation header.

### Cell 12 (code) - Train vs holdout metrics
Output: ROC AUC 0.8282 train versus 0.7600 holdout (KS 0.5083 vs 0.4011). The
gap is expected for a deep forest, but the holdout number is strong and the
best-F1 threshold is 0.45 (lower than the other families because the forest
probabilities are better calibrated).

### Cell 13 (code) - Holdout diagnostics
ROC/PR, calibration, score distribution, and confusion matrix.
Output: two figure groups; the calibration curve is notably closer to the
diagonal than the linear or tree models.

### Cell 14 (markdown) - Feature importance header.

### Cell 15 (code) - Feature importance
Top 20 importances for the forest.
Output: the importance figure.

### Cell 16 (markdown) - Takeaways
The forest sharply reduces single-tree variance, validation loss plateaus with
enough trees, and random forest and extra trees are close strong baselines
before gradient boosting.

## Key takeaways

- Holdout ROC AUC about 0.76, a clear step up from the single tree, with much
  better calibration (low Brier and log loss).
- The trees loss curve illustrates the core bagging idea: averaging more trees
  reduces variance without increasing overfitting.
- Still below the boosting family, which is tuned next.
