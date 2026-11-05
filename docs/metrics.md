# Discharge metrics

`hydrologeez.metrics.evaluate(observed, simulated, dates=None)` scores aligned
1-D discharge arrays. The optional keyword-only `dates` supplies daily NumPy
`datetime64` coordinates. Without dates, array positions are consecutive days.
Dates must be aligned, strictly increasing whole days without NaT; gaps are
allowed. Missing observations retain their calendar gaps for peak-window checks.
Calibration's `CandidateEvaluator` converts its ordinal dates to these daily
coordinates after applying the scoring mask, preserving excluded-day gaps.

## Metrics and objectives

The 14 shared metrics follow the executable defaults in
[NeuralHydrology revision 9528d1d13eb82c9f038ee489d43784bc7aeb08cc](https://github.com/neuralhydrology/neuralhydrology/blob/9528d1d13eb82c9f038ee489d43784bc7aeb08cc/neuralhydrology/evaluation/metrics.py).
Non-parametric KGE and log-NSE bring the suite to 16. In the table, `o` and `s`
are retained observations and simulations; means and standard deviations use
that complete scored period. Standard deviations use population normalization.

| Name | NeuralHydrology name / definition | Units | Optimum |
| --- | --- | --- | --- |
| `nse` | NSE: `1 - sum((s-o)²)/sum((o-mean(o))²)` | dimensionless | 1 |
| `mse` | MSE: `mean((s-o)²)` | discharge² | 0 |
| `rmse` | RMSE: `sqrt(mse)` | discharge | 0 |
| `kge` | KGE: `1 - sqrt((alpha-1)² + (beta-1)² + (pearson-1)²)` | dimensionless | 1 |
| `alpha` | Alpha-NSE: `std(s)/std(o)` | ratio | 1 |
| `beta` | Beta-KGE: `mean(s)/mean(o)` | ratio | 1 |
| `beta_nse` | Beta-NSE: `(mean(s)-mean(o))/std(o)` | dimensionless | 0 |
| `pearson` | Pearson-r: Pearson correlation | dimensionless | 1 |
| `high_flow_bias` | FHV: upper-tail volume bias | percent | 0 |
| `middle_flow_bias` | FMS: middle FDC log-slope bias | percent | 0 |
| `low_flow_bias` | FLV: anchored lower-tail log-shape bias | percent | 0 |
| `peak_timing` | Peak-Timing: mean absolute peak lag | days | 0 |
| `missed_peaks` | Missed-Peaks: fraction of observed peaks missed | fraction | 0 |
| `peak_mape` | Peak-MAPE: mean absolute percentage error at observed peaks | percent | 0 |
| `kge_np` | Pool (2018): `1 - sqrt((beta-1)² + (alpha_np-1)² + (spearman-1)²)` | dimensionless | 1 |
| `log_nse` | NSE of natural-log flows after adding `mean(o)/100` to both series | dimensionless | 1 |

`alpha_np` and `spearman` remain returned component diagnostics, not additional
objectives. `alpha_np = 1 - 0.5*sum(abs(sort(s)/sum(s)-sort(o)/sum(o)))`.
Spearman uses average ranks for ties. `alpha`, `beta`, and `pearson` are reused
as both components and selectable objectives, without duplicate names.
KGE uses unit weights. The reference docstring squares weighted deviations,
whereas its executable formula multiplies squared deviations by the weights;
this discrepancy does not affect the unit-weight calculation here.

`objective_loss(name, value)` minimizes `1-value` for efficiencies and Pearson,
`abs(value-1)` for `alpha` and `beta`, absolute value for signed FDC biases and
`beta_nse`, and the
value itself for MSE, RMSE, and peak metrics. Nonfinite scores yield infinity.
Choose metrics for the scientific question; overlapping objectives are allowed.

## Flow-duration conventions

Each series is sorted independently in descending order. Counts and indices use
NumPy rounding (nearest integer, ties to even), **not ceiling**. With `n` retained
days:

- FHV uses `[:round(0.02*n)]` and returns
  `100*sum(s_tail-o_tail)/sum(o_tail)`. An empty or zero-volume observed tail
  is undefined (NaN).
- FMS uses zero-based indices `round(0.2*n)` and `round(0.7*n)`.
  Replace zero flows with `1e-6` before taking natural logs. Let `D(Q)` be the
  lower-index log flow minus the upper-index log flow. Return
  `100*(D(s)-D(o))/(D(o)+1e-6)`. The reference's executable denominator is the
  **observed** slope; its docstring equation instead shows the simulated slope.
- FLV uses `[-round(0.3*n):]`, replacing zero flows with `1e-6` before logs.
  The literal negative-zero slice selects the whole array when the rounded
  count is zero. Let `A(Q)=sum(log(Q_tail)-min(log(Q_tail)))`. Return
  `-100*(A(s)-A(o))/(A(o)+1e-6)`.

Only exact zeros are replaced, after sorting; positive flows below `1e-6`
remain unchanged. The `1e-6` discharge replacement depends on flow units; it is not a universal
hydrological threshold. The added `1e-6` denominator adjustments regularize
FMS and FLV. Constant or all-zero tails have zero anchored shape: FLV is zero
when both shapes are zero, but can be very large when only the observed shape
is zero. Finite regularized scores do not prove useful discrimination in these
cases. Positive uniform rescaling leaves FLV unchanged when no zero replacement
is involved. These constants are NeuralHydrology conventions, not zero-handling
rules established by the Pool papers.

The log-NSE offset uses only retained observations and scales with flow units.
It follows hydroGOF's documented Pushpalatha convention, not log-KGE.

## Peak conventions

Peaks use SciPy `find_peaks` on observation-masked arrays. Detection distances
are sample counts after masking; dates are used to reject windows spanning
missing days, not to change the reference peak detector.

- Timing and peak MAPE detect observed peaks with distance 100 and prominence
  equal to the observed population standard deviation.
- Timing uses a ±3-day window. Peaks without a complete, consecutive window
  are skipped. If simulation at the observed peak is strictly above both its
  neighbors, timing error is zero; otherwise select the first maximum in the
  window and measure absolute lag. No eligible peaks gives NaN.
- Missed peaks detects peaks independently in each series with distance 30
  and height at least that series' 80th percentile. A simulated peak within
  ±1 day matches. Incomplete or gapped observed windows are skipped in the
  missed count, but remain in the denominator of all detected observed peaks,
  following the reference. No detected observed peaks gives zero (subject to
  the minimum-observation rule below).
- Peak MAPE compares simulation at the observed peak indices, without searching
  for shifted simulated peaks or applying timing-window exclusions. No detected
  peaks gives NaN.

## Input and undefined-value policy

Only NaN observations are excluded; observed zero is valid. Negative or infinite
observations raise an error. Negative or nonfinite simulations on scored days
invalidate all scores, not the observation mask. Missing-day simulations do not
affect scores. This is stricter than NeuralHydrology, which removes simulated
NaNs and replaces nonpositive simulated flows in FMS/FLV. Here negative flows
are never repaired into valid scores.

Fewer than two retained observations gives all NaNs. Constant correlation inputs
and unregularized zero denominators give NaN; identical constant series do not
have perfect KGE. FMS/FLV use the explicit regularizations above. Any nonfinite score cannot
reward an objective.

## Evaluate saved discharge

Use `hydrologeez.evaluation.evaluate_discharge` with existing
[output stores](hbv-output.md) and [prepared observations](camels.md):

```python
from pathlib import Path
from hydrologeez.evaluation import evaluate_discharge

scores = evaluate_discharge(
    Path("data/hbv-results"),
    Path("data/camels-us/prepared"),  # any location with the same prepared layout
    basin_ids=["03241500", "01013500"],
    parameter_ids=["p42", "p7"],
    periods=[("1995-01-01", "2000-01-01"), ("2000-01-01", "2005-01-01")],
)
for score in scores:
    print(
        score.basin_id,
        score.parameter_id,
        score.start,
        score.stop,
        score.observed_days,
        score.present_days,
        score.scores,
    )
```

Results follow basin, parameter, then period order. Each record contains all 16
metrics and their component diagnostics with the conventions above. The evaluator
passes the saved daily dates, retaining gaps caused by missing observations.
`observed_days` counts nonmissing converted observations; `present_days` counts
source rows, including rows with missing discharge. Period bounds identify the
number of requested days. No coverage threshold or QC filtering is applied.

Dates are start-inclusive and stop-exclusive. Observation dates outside their
recorded coverage are missing; the scoring period is never silently shortened.
Simulation dates must lie entirely within recorded output coverage, excluding
unsaved warm-up. Unknown identities, unavailable pairs, incomplete stores, and
incompatible units raise errors. Prepared cfs values use the existing header-area
conversion to mm/day; saved discharge must be mm/day.

This reads no forcing and runs no model. One simulation's full scoring period and
one basin's observation timeline are held at a time, rather than all simulation
histories. Scores are retained in a list; time windows are never scored separately
and averaged. Inputs must remain unchanged during evaluation.

## Sources

- [NeuralHydrology, pinned implementation](https://github.com/neuralhydrology/neuralhydrology/blob/9528d1d13eb82c9f038ee489d43784bc7aeb08cc/neuralhydrology/evaluation/metrics.py):
  shared metric formulas, rounding, numerical adjustments, and peak defaults.
- [Pool, Vis & Seibert (2018)](https://doi.org/10.1080/02626667.2018.1552002),
  equations 1–7, pp. 1943–1944; p. 1945 motivates absolute FDC bias.
- [Yilmaz et al. (2008)](https://doi.org/10.1029/2007WR006716): FDC diagnostics.
- [Gupta et al. (2009)](https://doi.org/10.1016/j.jhydrol.2009.08.003):
  MSE/NSE decomposition and KGE.
- [hydroGOF NSE documentation](https://search.r-project.org/CRAN/refmans/hydroGOF/html/NSE.html)
  documents the Pushpalatha (2012) observed-mean/100 offset. The primary
  Pushpalatha paper was not independently checked for this implementation.
