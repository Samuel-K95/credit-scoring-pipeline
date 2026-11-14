# Notebook 12 - Monitoring and drift

Source notebook: `notebooks/12_monitoring_drift.ipynb`

## Purpose

Show how the deployed model is monitored for population drift using the
Population Stability Index (PSI) on the score and the Characteristic Stability
Index (CSI) on the features. The notebook establishes a stable baseline (train
as reference, holdout as current), then simulates a drifting population to prove
the alerting bands fire. The thresholds follow the standard credit-risk
convention: PSI below 0.10 is stable, 0.10 to 0.25 is watch, and 0.25 or above
is an alert.

## Cell by cell

### Cell 0 (markdown) - Title and plan
Introduces PSI/CSI, the reference-versus-current framing, and the stable-then-
simulated-drift structure.

### Cell 1 (code) - Imports and data
Imports `population_stability_index`, `drift_status`, `feature_psi`, and
`score_psi` from `src.monitor`, loads the champion, and loads train and holdout.
Output: confirms the champion and the train/holdout shapes.

### Cell 2 (markdown) - Baseline header; notes train is the reference and holdout
is the current population, so PSI should be near zero because they come from the
same distribution.

### Cell 3 (code) - Baseline score PSI
Scores train and holdout with the champion and computes score PSI.
Output: score PSI about 0.00033, status stable - as expected for two samples of
the same population.

### Cell 4 (code) - Score distribution overlay
Overlays the train and holdout score histograms.
Output: the overlay figure; the two distributions sit almost exactly on top of
each other.

### Cell 5 (markdown) - Feature CSI header.

### Cell 6 (code) - Baseline feature CSI
`feature_psi` across all features, reference train vs current holdout.
Output: a sorted table; every feature is comfortably below 0.10, confirming no
baseline drift.

### Cell 7 (markdown) - Simulated drift header; explains the current population is
perturbed (shift key drivers, inflate late-payment rates) to mimic a worsening
book.

### Cell 8 (code) - Build a drifted population
Copies the holdout and shifts several drivers (lower external sources, higher
recent late-instalment rates).
Output: a short note on which features were perturbed and by how much.

### Cell 9 (code) - Score PSI under drift
Recomputes score PSI against the drifted population.
Output: score PSI about 0.92796, status alert - the score distribution has moved
dramatically.

### Cell 10 (code) - Feature CSI under drift
Recomputes feature CSI against the drifted population.
Output: the sorted table flags 6 features at alert and 1 at watch, led by
inst_late_rate_6m (CSI about 12.4) and the shifted external sources; the
untouched features stay stable, so the index correctly localises the drift.

### Cell 11 (markdown) - Takeaways
Summarises that score PSI is the top-line monitor, feature CSI localises the
cause, the bands match `src/config.py` (PSI_STABLE_MAX 0.10, PSI_WATCH_MAX
0.25), and `src/monitor.py` produces `reports/drift_report.md` and
`reports/drift.json` from the same functions.

## Key takeaways

- Baseline score PSI about 0.0003 (stable) validates the monitor on same-
  population data.
- Simulated drift drives score PSI to about 0.93 (alert) and flags the exact
  perturbed features via CSI, led by inst_late_rate_6m.
- The notebook mirrors the runnable `src/monitor.py` CLI, whose thresholds come
  from `src/config.py`.
