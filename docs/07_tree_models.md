# Notebook 07 - Tree models (decision tree)

Source notebook: `notebooks/07_tree_models.ipynb`

## Purpose

Fit a single CART decision tree: non-linear and highly interpretable, but prone
to overfitting. The notebook tunes complexity controls with grid search, shows
the classic depth-versus-loss overfitting curve, evaluates on the holdout, and
inspects feature importance. Same leakage-safe pipeline pattern as notebook 06.

## Cell by cell

### Cell 0 (markdown) - Title and plan
Lists the steps: tune depth, leaf size, criterion, class weight; plot the depth
loss curve and CV validation curve; evaluate on holdout and show importance.

### Cell 1 (code) - Imports and data
Same toolkit imports as notebook 06 and `load_modeling_data()`.
Output: the data summary (246005 train, 61502 holdout, 52 features, 0.0807 rate,
5 folds, pos_weight 11.39).

### Cell 2 (code) - Search sample
`SEARCH_FRAC = 0.3` builds a 30 percent stratified subsample.
Output: the sample summary - 73802 train rows.

### Cell 3 (markdown) - Hyperparameter search header.

### Cell 4 (code) - Grid search
`GridSearchCV` over `DecisionTreeClassifier` with `max_depth` in
{3, 4, 6, 8, 10}, `min_samples_leaf` in {50, 100, 200}, `criterion` in
{gini, entropy}, class weight balanced, refit on ROC AUC.
Output: 30 candidates times 5 folds, best ROC AUC 0.7070 with entropy,
max_depth 8, min_samples_leaf 200. The results table shows the key tension:
deeper trees have much higher train AUC (for example 0.776 at depth 10) but the
validation AUC stays around 0.70, which is the overfitting signal that favors a
shallow, well-regularised tree.

### Cell 5 (markdown) - Refit header.

### Cell 6 (code) - Refit on full train
Refits the selected tree on all training rows.
Output: the selected params and the fitted Pipeline (nested preprocess plus the
entropy depth-8 tree).

### Cell 7 (markdown) - Depth loss curve header; explains that a single tree has
no iterations, so `max_depth` is swept and train/validation log loss recorded,
with the validation minimum marking the onset of overfitting.

### Cell 8 (code) - Depth loss curve
Uses `param_loss_curve` over depths 1 to 16 and plots train versus validation
log loss.
Output: the depth loss-curve figure, where validation loss falls then rises as
depth grows, marking the overfitting point.

### Cell 9 (markdown) - CV curves header.

### Cell 10 (code) - Validation and learning curves
Validation curve over `max_depth` and a learning curve on the search sample.
Output: two figures.

### Cell 11 (markdown) - Holdout evaluation header.

### Cell 12 (code) - Train vs holdout metrics
Output: a visible train-to-holdout gap (ROC AUC 0.7470 train vs 0.7162 holdout,
KS 0.3657 vs 0.3328), showing the tree overfits more than the linear model even
after tuning. Best-F1 threshold 0.65.

### Cell 13 (code) - Holdout diagnostics
ROC/PR, calibration, score distribution, and confusion matrix at the best-F1
threshold.
Output: two figure groups.

### Cell 14 (markdown) - Feature importance header.

### Cell 15 (code) - Feature importance
Plots the top 20 impurity-based importances for the tree.
Output: the importance figure (external scores and the affordability ratios
dominate, consistent with the other families).

### Cell 16 (markdown) - Takeaways
The depth loss curve shows the overfitting point; CV picks a shallow tree; a
single tree trails logistic regression on AUC but captures interactions that
ensembles and boosting then exploit.

## Key takeaways

- Holdout ROC AUC about 0.716, slightly above the linear baseline but with a
  wider train-to-holdout gap.
- The depth loss curve is the headline diagnostic: it visualizes exactly where
  a deeper tree stops generalizing.
- A single tree is mainly a stepping stone to the ensemble and boosting
  families.
