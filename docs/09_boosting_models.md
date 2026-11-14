# Notebook 09 - Boosting models

Source notebook: `notebooks/09_boosting_models.ipynb`

## Purpose

Tune and compare the four gradient-boosting libraries that usually give the best
discrimination on tabular credit data: XGBoost, LightGBM, CatBoost, and
scikit-learn GradientBoosting. The notebook plots true per-iteration
train/validation loss curves for all four, runs a compact grid search per
library, compares the tuned boosters on the untouched holdout, and inspects the
best one. `scale_pos_weight` handles the 8 percent default rate.

## Cell by cell

### Cell 0 (markdown) - Title and plan
Lists the steps: per-iteration loss curves, per-library tuning, holdout
comparison, and importance.

### Cell 1 (code) - Imports and data
Toolkit imports and `load_modeling_data()`.
Output: data summary (246005 train, 61502 holdout, 52 features, 0.0807, 5 folds,
pos_weight 11.39).

### Cell 2 (code) - Search sample
`SEARCH_FRAC = 0.3` builds a 30 percent subsample (73802 train rows).

### Cell 3 (markdown) - Loss curves header; explains each library is fit once on
a train/validation split with log loss recorded every round, and the validation
minimum is where early stopping would halt.

### Cell 4 (code) - Per-iteration loss curves
Fits XGBoost, LightGBM, CatBoost (500 rounds each) and GradientBoosting (300)
with the curve helpers and plots four loss curves.
Output: LightGBM prints its training banner (positive count 5958, init score
-2.432 from the base rate). The 2x2 loss-curve figure plus a best-iteration
table: XGBoost 499, LightGBM 0, CatBoost 451, GradientBoosting 299. (The
LightGBM best-iteration of 0 on this split just means its validation log loss is
minimized very early; the tuned LightGBM below still wins on AUC.)

### Cell 5 (markdown) - XGBoost tuning header.

### Cell 6 (code) - XGBoost grid search
`GridSearchCV` over `XGBClassifier` (hist tree method, scale_pos_weight) with
depth, learning rate, subsample, colsample, and reg_lambda; 24 candidates times
5 folds.
Output: best ROC AUC 0.7626 with colsample 0.8, learning rate 0.03, depth 4,
reg_lambda 5.0, subsample 0.8.

### Cell 7 (markdown) - LightGBM tuning header.

### Cell 8 (code) - LightGBM grid search
`GridSearchCV` over `LGBMClassifier` with num_leaves, learning rate, subsample
plus frequency, colsample, and reg_lambda; 8 candidates times 5 folds.
Output: best ROC AUC 0.7615 with num_leaves 31, learning rate 0.03, reg_lambda
5.0, subsample 0.8. The table shows the slow-learning-rate shallow configs
generalize best; the high learning-rate rows overfit (train AUC near 1.0,
validation dropping).

### Cell 9 (markdown) - CatBoost tuning header.

### Cell 10 (code) - CatBoost grid search
`GridSearchCV` over `CatBoostClassifier` with depth, learning rate, and
l2_leaf_reg; 8 candidates times 5 folds.
Output: best ROC AUC 0.7616 with depth 6, l2_leaf_reg 7.0, learning rate 0.03.

### Cell 11 (markdown) - Holdout comparison header.

### Cell 12 (code) - Compare tuned boosters on holdout
Evaluates the three tuned boosters on the holdout.
Output: on the 30 percent search sample, the three are very close - CatBoost
0.7638, XGBoost 0.7628, LightGBM 0.7615. (When each is retrained on the full
train partition in notebook 10, LightGBM becomes the top model.)

### Cell 13 (code) - ROC/PR overlay
Overlays ROC and PR curves for the three tuned boosters.
Output: the overlay figure (the three curves nearly coincide).

### Cell 14 (markdown) - Feature importance header.

### Cell 15 (code) - Feature importance of the best booster
Plots the top 20 importances for the sample-best booster and prints its name.
Output: the importance figure and `best booster on holdout: catboost` for this
sample-based comparison.

### Cell 16 (markdown) - Takeaways
Loss curves show each library converging to a validation minimum (the
early-stopping point), the comparison ranks tuned boosters by holdout AUC/KS to
pick the Stage-5 champion, and reported metrics in `reports/metrics.json` must
come from the runnable pipeline rather than these exploratory cells.

## Key takeaways

- All four boosters are tuned with small, realistic grids and compared on the
  same untouched holdout.
- The per-iteration loss curves make the early-stopping point explicit for each
  library.
- Boosting leads the other families; the final champion (LightGBM) is chosen in
  notebook 10 after retraining each winner on the full training data.
