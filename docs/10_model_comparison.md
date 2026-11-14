# Notebook 10 - Model comparison and champion selection

Source notebook: `notebooks/10_model_comparison.ipynb`

## Purpose

Bring together the tuned champion from every family (notebooks 06 to 09),
retrain each on the full training partition, evaluate them all on the same
untouched holdout, compare discrimination and calibration, and pick the single
best model. The tuned configurations live in `src.modeling.champions` and fitted
models are cached in `models/` through `src.modeling.registry`, so the first run
trains and saves each model while later runs just load them.

## Cell by cell

### Cell 0 (markdown) - Title and plan
Describes the apples-to-apples comparison across the six family champions:
logistic regression (linear), decision tree (tree), random forest (ensemble),
and XGBoost/LightGBM/CatBoost (boosting).

### Cell 1 (code) - Imports and data
Imports the champions factory, metrics, registry, and plots, and loads the
modelling data.
Output: the data summary (246005 train, 61502 holdout, 52 features, 0.0807 rate,
5 folds, pos_weight 11.39).

### Cell 2 (markdown) - Load-or-train header; notes each champion is a full
pipeline trained on the entire training partition, cached models are loaded, and
the first run can take 10 to 20 minutes.

### Cell 3 (code) - Load or train each champion
Builds all champions with the shared positive-class weight, then for each either
loads the cached model or trains it on the full train partition, evaluates on
the holdout, and saves it to the registry.
Output: training log with per-model timing and holdout AUC - xgboost 36.1s
(0.7681), catboost 55.4s (0.7675), lightgbm 29.0s (0.7725), random_forest 438.5s
(0.7647), logistic_regression 90.0s (0.7128), decision_tree 17.1s (0.7162).
These are the full-train numbers, which differ from the sample numbers in
notebook 09.

### Cell 4 (markdown) - Comparison table header.

### Cell 5 (code) - Holdout comparison table
Builds a metrics frame for all six models on the holdout, adds family and
search-sample CV AUC columns, and sorts by ROC AUC.
Output: the ranking - lightgbm 0.7725 (Gini 0.5451, KS 0.4171), xgboost 0.7681,
catboost 0.7675, random_forest 0.7647, decision_tree 0.7162, logistic_regression
0.7128. Random forest has the best log loss and Brier (best calibrated), but
LightGBM leads on discrimination.

### Cell 6 (markdown) - ROC/PR overlay header.

### Cell 7 (code) - ROC and PR overlay
Overlays all six models' ROC and PR curves on the holdout.
Output: the overlay figure, where the three boosters and the forest cluster
above the tree and linear models.

### Cell 8 (markdown) - Calibration overlay header.

### Cell 9 (code) - Calibration overlay
Overlays reliability curves for all six.
Output: the calibration figure (the forest hugs the diagonal; the boosters,
trained with scale_pos_weight, are less calibrated but rank well).

### Cell 10 (markdown) - Bar charts header.

### Cell 11 (code) - AUC and KS bar charts
Two horizontal bar charts ranking the models by ROC AUC and by KS.
Output: the bar-chart figure.

### Cell 12 (markdown) - Select and persist header; notes the champion is tagged
in the registry and a summary is written to `reports/model_comparison.json`, and
that the official `reports/metrics.json` comes from the Stage 6 pipeline, not
this notebook.

### Cell 13 (code) - Select and persist the champion
Takes the top model by holdout AUC, saves it as `champion` in the registry, and
writes the comparison summary JSON.
Output: `champion: lightgbm ( boosting )`, holdout AUC 0.7725, KS 0.4171, and
the written path. This is the point where LightGBM becomes the project champion.

### Cell 14 (markdown) - Takeaways
The comparison is on the same untouched holdout, boosting leads with logistic
regression as the interpretable baseline, and the champion is cached for the
Stage 6 pipeline.

## Key takeaways

- Final ranking on the full-train holdout: LightGBM > XGBoost > CatBoost >
  Random Forest > Decision Tree > Logistic Regression.
- LightGBM is selected as champion (ROC AUC 0.7725, Gini 0.545, KS 0.417).
- The notebook populates the model registry and writes
  `reports/model_comparison.json`, which `src/train.py` then reads to train the
  same champion in the runnable pipeline.
