# Deterministic discharge metric suite

Expand discharge evaluation from six metrics to the 14 deterministic metrics in NeuralHydrology, retaining non-parametric KGE and log-NSE for a total of 16. Use its established numerical conventions rather than inventing a new low-flow metric. This is a tooling-only effort.

This vision expands [issue #164](https://github.com/CooperBigFoot/hydrologeez/issues/164). Implementing and verifying this vision completes that issue; publishing this document does not.

## Metric scope

| NeuralHydrology metric | Purpose |
| --- | --- |
| NSE | Nash–Sutcliffe efficiency |
| MSE | Mean squared error |
| RMSE | Root mean squared error |
| KGE | Original Kling–Gupta efficiency |
| Alpha-NSE | Standard-deviation ratio |
| Beta-NSE | Mean error normalized by observed standard deviation |
| Beta-KGE | Simulated/observed mean ratio |
| Pearson-r | Pearson correlation |
| FHV | High-flow flow-duration-curve bias |
| FMS | Middle-segment flow-duration-curve slope bias |
| FLV | Low-flow flow-duration-curve bias |
| Peak-Timing | Mean peak timing error |
| Missed-Peaks | Fraction of missed peaks |
| Peak-MAPE | Mean absolute percentage peak error |

Also retain `kge_np`, the Pool, Vis & Seibert (2018) non-parametric KGE, and `log_nse`, with its current observed-mean/100 offset. Existing component diagnostics remain available; overlap between existing components and the expanded suite does not require duplicate computations or redundant names.

## Numerical reference and repository contract

Use [NeuralHydrology metrics.py at revision 9528d1d13eb82c9f038ee489d43784bc7aeb08cc](https://github.com/neuralhydrology/neuralhydrology/blob/9528d1d13eb82c9f038ee489d43784bc7aeb08cc/neuralhydrology/evaluation/metrics.py) as the pinned reference for the 14 metrics. Follow its executable formulas, defaults, and numerical conventions on valid inputs, including tail selection, peak detection and matching, and zero/denominator adjustments. Inspect the actual implementation rather than relying only on docstring equations. Record any source discrepancy that affects interpretation.

In particular, the reference FLV replaces zero discharge with `1e-6` before logarithms and adds `1e-6` to the observed log-shape denominator. Its default low-flow fraction is 30%. These are implementation conventions, not a zero-handling rule established by the Pool papers. Document the flow-unit dependence of a fixed discharge replacement and the behavior of constant/all-zero tails. Do not describe finite regularized results as proof that every degenerate case has meaningful hydrological discrimination.

Follow hydrologeez standards rather than importing NeuralHydrology's architecture:

- Preserve existing public metric names and the aligned 1-D evaluation entry point. Leave the exact organization and naming of additions to implementation.
- Preserve the observation mask: exclude only missing observations; observed zero is valid. Negative or infinite observations remain errors. Invalid simulations on scored days must not silently remove days or be repaired into valid flows. Missing-day simulations do not affect scores.
- Preserve the current minimum-observation rule and nonfinite-objective handling. Undefined cases must be explicit, not silently rewarded.
- Integrate the suite with saved-discharge evaluation and existing objective handling. Give each exposed objective its correct direction and optimum; do not assume all non-bias metrics have optimum one. Existing outputs and component diagnostics should remain coherent.
- Account for peak metrics' time-coordinate requirements in both direct-array and dated saved-discharge evaluation. Keep existing callers working, make any daily-spacing assumption explicit, and do not silently compress missing-day gaps when measuring timing. Resolve the smallest suitable interface during implementation.
- Use strongly typed Python, `uv`, Ruff, ty, and repository-native tests. Reuse suitable dependency utilities where warranted; a production dependency on NeuralHydrology is not required.

Where the stricter input contract differs from NeuralHydrology, document the difference and test it separately from numerical parity. Existing numerical conventions such as ceiling-based tail counts may change to match the pinned reference; retaining names does not require retaining conflicting calculations.

## Evidence and verification

The present implementation in `src/hydrologeez/metrics.py` uses the anchored logarithmic FLV formula without a floor or denominator adjustment. It returns NaN when either scored series contains zero, and when the observed low tail is constant. Positive uniform rescaling leaves its FLV unchanged. These behaviors were reproduced with small arrays; the issue's reported 133-basin result was not independently rerun for this vision.

Pool et al. (2018), p. 1945, refers its low-flow bias to Yilmaz et al. (2008) without specifying zero handling. Pool et al. (2017), Table 1, instead includes the E85 percentile signature; Pool et al. (2021) uses KGE/non-parametric KGE and Q5 among catchment descriptors. Neither supplies the missing FLV zero convention. Cite NeuralHydrology for implementation conventions and the original publications for metric definitions, rather than attributing every adjustment to those papers.

Success means:

- All 16 metrics are available through the tooling, with existing names preserved and saved-discharge evaluation covered.
- Focused comparison tests demonstrate numerical agreement with the pinned reference for the 14 shared metrics under equivalent valid-input conditions. Use independent expected values for focused edge cases; avoid a second hand-written implementation as the sole oracle.
- FLV coverage includes observed-only zeros, simulated-only zeros, both-series zeros, all-zero tails, and constant tails. Retain coverage of invalid inputs and missing observations, and test distinct peak/timing behaviors and objective directions.
- Documentation in `docs/metrics.md` explains the metric mapping, units, optima, numerical constants, defined/undefined cases, input-policy differences, and reference revision. Update contradictory existing statements about FLV and tail selection.
- Relevant tests, Ruff, and ty pass in the project environment.

## Boundaries

Do not add CRPS, prediction interval coverage, prediction interval width, or an ensemble input interface. Do not change experiments, candidate acceptance rules, scientific thresholds, or model simulations. No separate dry-day metric or replacement low-flow definition is requested. Implementation details remain delegated within the contracts above.
