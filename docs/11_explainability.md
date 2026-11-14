# Notebook 11 - Explainability (SHAP)

Source notebook: `notebooks/11_explainability.ipynb`

## Purpose

Explain the LightGBM champion with SHAP values so the scorecard can produce
honest per-applicant reason codes. The notebook uses LightGBM's native tree SHAP
(`booster_.predict(X, pred_contrib=True)`) rather than the external `shap`
package, which keeps the dependency footprint light and still reconstructs the
model output to machine precision. SHAP values are in margin (log-odds) space,
so they add up to the raw score before the sigmoid.

## Cell by cell

### Cell 0 (markdown) - Title and plan
Explains global importance, local explanations, dependence, and how these map to
reason codes; notes SHAP values are additive in log-odds space.

### Cell 1 (code) - Imports and data
Imports `compute_shap` and the plotting helpers from `src.modeling.explain`,
loads the champion from the registry, and loads the holdout.
Output: confirms the loaded champion (lightgbm) and the holdout shape.

### Cell 2 (markdown) - Sampling header; notes a few thousand rows is enough for
stable global SHAP and keeps plots responsive.

### Cell 3 (code) - Sample and compute SHAP
Samples about 4000 holdout rows and calls `compute_shap`, which returns a
`ShapResult` with values, feature names, and the base value.
Output: the SHAP matrix shape (about 4000 x 52) and the base value.

### Cell 4 (markdown) - Reconstruction check header.

### Cell 5 (code) - Reconstruction check
Confirms that base value plus the row's SHAP values, pushed through the sigmoid,
equals the model's `predict_proba`.
Output: max reconstruction error about 2.78e-15 - machine precision, proving the
native SHAP path is exact.

### Cell 6 (markdown) - Global importance header.

### Cell 7 (code) - Global importance bar
`plot_global_bar` on mean absolute SHAP.
Output: the global bar; top drivers are app_ext_source_2 (mean abs SHAP about
0.368), app_ext_source_3, app_ext_source_1, and the external-source aggregates,
matching domain expectation that the bureau scores dominate.

### Cell 8 (markdown) - Beeswarm header.

### Cell 9 (code) - Beeswarm
`plot_beeswarm` colouring points by feature value (red high, blue low).
Output: the beeswarm; for the external sources, high values (red) sit on the
negative-SHAP side, i.e. a strong bureau score pushes risk down, as expected.

### Cell 10 (markdown) - Dependence header.

### Cell 11 (code) - Dependence plots
`plot_dependence` for the top few drivers.
Output: dependence figures showing the monotone-ish relationship between each
driver and its SHAP contribution.

### Cell 12 (markdown) - Local explanations header; explains a waterfall walks
from the base value to one applicant's score.

### Cell 13 (code) - Waterfall for highest and lowest risk
Picks the highest- and lowest-risk holdout applicants and draws a waterfall for
each.
Output: two waterfalls; the high-risk case is driven up by weak external sources
and recent late instalments, the low-risk case down by strong external sources.

### Cell 14 (markdown) - Takeaways
Explains how global importance justifies the feature set, local SHAP drives
reason codes in `src/scoring.py`, and the reconstruction check proves fidelity.

## Key takeaways

- Native LightGBM SHAP reconstructs `predict_proba` to about 2.78e-15, so the
  explanations are exact, with no heavy extra dependency.
- External-source features dominate global importance; strong bureau scores lower
  predicted risk.
- The same `compute_shap` function powers the production reason codes in
  `src/scoring.py` and the report produced by `src/explain.py`.
