# Joint multi-basin HBV calibration

Support calibration of shared HBV parameter vectors against an explicitly selected
collection of basins. Each candidate vector applies to every selected basin;
calibration may retain multiple such vectors. This is research tooling, not an
experiment or a claim that shared parameters perform well or generalize.

Implementing this vision completes [issue #167](https://github.com/CooperBigFoot/hydrologeez/issues/167).
Publishing this document does not complete the issue.

## Evaluation and optimization

Run each candidate on each selected basin with independent hydrological states,
warm-up, and basin-specific observation masks. Share parameters, not water stores.
Do not pool observations across basins. Compute each metric over its full scoring
period rather than averaging metrics computed on time windows.

Convert each basin's metric to a minimization loss before aggregating:

- Efficiencies and correlation: `1 - score`.
- Ratios `alpha` and `beta`: `abs(score - 1)`.
- Signed biases, including `beta_nse`: `abs(score)`.
- MSE, RMSE, and peak-error metrics: minimize the value.

Reuse the existing metric-direction conventions in `objective_loss()`. In
particular, opposite signed biases must not cancel across basins.

For joint calibration, normalize each basin's RMSE by its mean observed training
flow and MSE by the square of that mean. Compute the mean from observations retained
by that basin's training mask. These observation-only denominators remain fixed
throughout calibration; do not derive them from candidate scores or validation
observations. If reporting normalized validation losses, use the same training
denominators. Other metrics retain their existing scales, including peak timing
in days. Do not change the raw metric definitions or their diagnostic values.

For each selected metric, minimize the weighted mean of these basin losses.
Weights default to equal basin weights. Accept explicit positive finite basin
weights, normalize them to sum to one, and use the same weights for every objective.
Do not automatically weight by record length, basin area, or discharge magnitude.
Keep different metrics as separate optimization objectives; do not average them
together. Support the existing single-objective and multi-objective search modes.

Only weighted-mean aggregation is in scope. Do not add custom aggregation callbacks,
worst-basin objectives, threshold objectives, or a framework for future aggregators.

## Validity and results

Every selected basin must have valid losses for every selected optimization metric.
Never silently omit a basin or renormalize weights around undefined scores.
Reject observation-side incompatibilities before optimization where detectable,
such as zero mean flow for a requested normalized error or constant observations
for a requested correlation objective. Treat candidate-dependent undefined losses
as invalid candidates. Undefined diagnostic-only metrics may remain `NaN`.

Retain raw per-basin diagnostics alongside aggregated optimization losses, with
unambiguous basin and parameter identities and reproducible run seeds. Make the
selected objectives, weights, and normalization clear enough to interpret and
reproduce the aggregate. Do not label minimized or normalized losses as raw metric
scores.

Preserve the existing single-basin calibration use case and its metric semantics.
Bound simulation memory through batching rather than materializing all candidate,
basin, and time combinations at once. Exact public API structure, result types,
and batching strategy remain implementation choices; keep the interface small.

## Repository context and evidence of success

`src/hydrologeez/calibration/__init__.py` currently provides a single-basin
`CandidateEvaluator`, seeded calibration, and retained solutions. Multiple
objectives currently mean multiple metrics for that one basin. The HBV simulator
already supports multi-basin execution. Metric formulas and minimization mappings
are in `src/hydrologeez/metrics.py` and [the metric documentation](../../docs/metrics.md).

Demonstrate that joint losses match independently evaluated basin results after
the specified loss conversion, normalization, and weighting. Cover metric
direction and target-value handling, normalization with fixed masked training
means, explicit and equal weights, and undefined-objective behavior. Verify that
independent states and masks, multi-objective results, reproducible identities and
seeds, and bounded batching behave as intended. Reuse existing coverage where it
already establishes these promises, including single-basin behavior.

Document a small joint-calibration example and the aggregation semantics. State
that optimizing the aggregate does not guarantee acceptable performance in every
basin. No regional experiment, cluster selection, behavioral threshold, scientific
comparison, or regionalization claim is part of this work. Follow repository
practice: strongly typed Python, uv, Ruff, ty, and test-first development.
