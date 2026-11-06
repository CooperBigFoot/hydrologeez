# Saved discharge evaluation

Connect observation reading and saved-discharge reading to the existing scoring
backend so large saved runs can be evaluated without calibration or model reruns.
This is backend capability that future scripts can call; implementing or running
those scripts is not part of this work.

This vision covers [#148](https://github.com/CooperBigFoot/hydrologeez/issues/148).
Implementing and verifying it completes that issue; publishing this document does not.

## Behavior

- Accept paths to saved model outputs and prepared observations using the existing
  physical layouts. A caller may supply a different prepared-data location with
  the same layout; do not introduce a new observation format or source framework.
- Support selecting basins, parameter sets, and explicit scoring date ranges.
  Keep the existing start-inclusive, stop-exclusive date convention.
- Align observations and simulated discharge by basin identity and date, not
  incidental file or column order. Check matching units and respect recorded
  coverage. Warm-up excluded from saved outputs must not enter scoring.
- Compute all six existing metrics for each basin × parameter set × scoring
  period: non-parametric KGE, original KGE, NSE, log-NSE, high-flow bias, and
  low-flow bias. Preserve existing missing-observation, invalid-simulation, and
  undefined-score conventions; do not duplicate or change metric formulas.
- Expose the scores with their basin, parameter, and scoring-period identities
  and observation coverage. Do not add scientific coverage thresholds or QC rules.
- Bound live discharge data across large collections of simulations. Evaluate
  each complete scoring period; averaging independently scored time windows is
  not equivalent. Full-period data for a bounded subset of simulations is acceptable.

## Existing pieces to connect

[`OutputStore.read`](../../src/hydrologeez/hbv_light/output.py) reads selected
variables, basin/parameter pairs, and dates from complete output stores. Its
selection carries dates and variable metadata; discharge is in mm/day. Reads
reject unavailable pairs and dates outside recorded coverage. See
[output storage](../../docs/hbv-output.md).

[`Camels.observations`](../../src/hydrologeez/camels.py) reads a basin's observation
timeline independently of forcing selection. It exposes dates, discharge in cfs
and mm/day, QC strings, and source-row presence. The existing conversion uses the
prepared basin's forcing-header area. Reuse the prepared-data contract rather
than loading forcing merely to obtain observations. See
[prepared inputs](../../docs/camels.md).

[`hydrologeez.metrics.evaluate`](../../src/hydrologeez/metrics.py) already scores
aligned one-dimensional arrays. Missing observations are excluded; invalid
simulations on scored days and undefined individual metrics retain their existing
behavior. See [metric definitions and conventions](../../docs/metrics.md).

## Scope and implementation freedom

No CLI, runnable experiment scripts, agent workflow, calibration integration,
model changes, observation acquisition, or QC implementation. QC before an
experiment is outside this work. Do not redesign existing output/state/flux
saving or introduce mandatory score persistence or a new storage workflow.

Choose a small typed backend interface and result representation. Exact function
signatures, module placement, and batching mechanics are implementation choices,
not reasons to build a general evaluation framework.

## Evidence of completion

Document a concise backend usage example from existing saved outputs and prepared
observations to identified scores. Test the connected path against direct array
calls to the existing scoring backend for all six metrics, including reordered
basins/parameters, explicit date ranges, missing observations, and undefined
scores. Verify that recorded coverage and units are respected, no model run is
needed, and processing does not materialize all simulation histories at once.
Use the project's uv, pytest, Ruff, and ty workflow.
