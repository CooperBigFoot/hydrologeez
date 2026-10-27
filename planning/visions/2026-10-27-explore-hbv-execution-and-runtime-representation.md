# Explore HBV execution and runtime representation

## Outcome and authority

Produce evidence for choosing a fast, faithful HBV implementation for thousands of
basins and many parameter sets per basin. This is the experimentation branch
[#99](https://github.com/CooperBigFoot/hydrologeez/issues/99) of
[implementation discussion #91](https://github.com/CooperBigFoot/hydrologeez/issues/91),
within [Program #87](https://github.com/CooperBigFoot/hydrologeez/issues/87).

Completion delivers runnable experiments and an evidence report to #99, with
findings linked back to #91. It completes only the experiment issue. It does not
select or ship a production implementation and does not complete #91. The user
will resume the grill in #91, interpret the evidence, choose an approach, and
request a separate implementation vision. Publication alone completes neither
issue and starts no experiments.

This supersedes the earlier discussion direction that the immediate work would
both explore alternatives and ship a selected implementation. Recommendations
are welcome; production adoption remains a later decision.

## Behavioral and usage anchors

Use [the documented daily HBV model](../../docs/hbv-light.md) as the behavioral
reference and [the mathematical framework](../../docs/mathematical-framework.md)
as the description philosophy. Preserve process laws, daily ordering, thresholds,
available-water allocation, area-weighted transfers, output timing, and complete
restart memory. Do not claim verified equivalence to an HBV-light executable.

The [model-configuration vision](2026-10-27-bucket-model-configuration-surface.md)
anchors intended use, not configuration machinery. In particular:

- Snow and soil memory belongs to zones; groundwater and routing are basin-wide.
- Snow/soil parameters can be shared per basin, elevation band, vegetation class,
  or zone, independently per parameter. Sharing values never merges state.
  This extends the parameterization described in the HBV document, not its
  process equations; state that distinction in experiment documentation.
- Initially all parameters may vary between candidates. Selected subsets may
  later be fixed, including different subsets for different basins. Fixed values
  remain explicit inputs, not hard-coded scientific constants.
- Many candidates reuse the same basin forcings. One candidate simulation is
  not one calibration. Calibration algorithm selection and orchestration are
  outside this experiment.

No parser, TOML loader, external parameter-table resolver, sampling framework,
or configuration-driven workflow is required. Model calculations know nothing
about configuration files. Experimental callers supply resolved inputs directly.

## Inputs and separation of responsibilities

Use reproducible synthetic inputs: prepared daily precipitation, temperature,
and PET; basin/zone identities and area fractions; parameter-sharing maps;
resolved candidate values; and explicit initial memory. Document units, time
alignment, parameter domains, and initial conditions. Do not invent scientific
defaults from illustrative configuration values.

Offline preparation could eventually derive zones from DEM and vegetation data,
prepare forcings, and save them for reuse. It is outside this work. No raster
processing, real-data ingestion, final storage contract, or public data API is
required. HDX and HFX were examples of explicit contracts, not dependencies or
formats selected for hydrologeez. Coordinate useful reader/layout findings with
[#90](https://github.com/CooperBigFoot/hydrologeez/issues/90), without implementing
that effort here.

Define enough experimental input semantics to compare the same simulations
across implementations. Physical array ordering, batching, padding/packing,
precision, and device representation are experiment variables, not a frozen API.
Include multi-zone cases and differing basin zone counts; do not base conclusions
only on identical single-zone basins.

## Experimentation

Start with a few plausible alternatives, measure them, and iterate from the
findings. Compare against a strong compiled sequential-time baseline, not only
an interpreted Python timestep loop. No backend or performance winner is chosen.
The implementing agent may work sequentially or in dedicated worktrees under
repository checkout policy. An exhaustive tournament or general benchmark
framework is not required.

Investigate execution and memory representation together. Candidate ideas include
parallelism over independent basin–parameter simulations, forcing reuse, fixed-
parameter precomputation, alternative routing evaluation, and compatible temporal
methods. These are hypotheses, not a mandatory checklist or chosen architecture.
Nonlinear daily rules do not automatically admit an affine scan or fixed linear
convolution. Use structure actually present in this model.

Only reuse calculations when their dependencies permit it. For example, fixing
basin data permits reuse across candidates; varying snow parameters can invalidate
precomputed snow results. Candidate parameters also apply through warm-up: do not
silently reuse warm-up state produced by different parameters.

Keep experiments bounded to the local Mac CPU and GPU where supported. Report
hardware and backend limits; no NVIDIA rental or remote compute is authorized.
Use project-native execution and dependencies through uv, typed Python where
Python is used, and the repository's Ruff, ty, and test-first guidance. Do not
force alternatives into a common internal architecture merely to simplify comparison.
A routines-based source layout is a useful starting point for paper readers,
not a plug-in system or execution boundary; revise it if it causes friction.

## Correctness before performance claims

Use controlled cases with independently established expectations, including the
document's worked day and meaningful boundary cases. Establish correctness beyond
water balance alone: compare predictions, states and transfers, zone aggregation,
output timing, and restart continuation including routing memory.

Separate equivalent reformulations and floating-point differences from changes
to model behavior. Altering daily priorities, smoothing thresholds, or replacing
the specified routing law is not a faithful optimization. Numerical comparisons
must state precision, reference, error measures, and justified tolerances. An
inaccurate contender may be reported as a finding but not ranked as a faithful
speedup. Synthetic results do not establish real-world hydrological skill.

## Predictions, histories, and output costs

Expose timestep outlet predictions and optionally record full timestep state
histories, including routing memory. Preserve access to the other model quantities
specified by the configuration vision when requested. Record the initial state
and make start/end-of-day state timing explicit. Associate outputs with their
basin, parameter set, zone where applicable, and timestep.

Recording is selectable through code. Do not require retaining all simulations'
full histories in memory. A simple batched Parquet writer in experimental output
code is sufficient for disk recording; model calculations remain independent of
that writer. Parquet is not selected as a permanent storage contract, and a public
streaming API is not required. Keep bounded-memory recording and complete restart
state possible without designing a general output system.

Compare equivalent requested outputs. Include prediction-only and state-history
recording workloads. Distinguish calculation, history materialization, and disk
writing costs so deferred work is not presented as a speedup.

## Evidence to return

Use shared, seeded workloads with enough variation to exercise snow accumulation
and melt, wet/dry soil, groundwater recession, and routing. Include workload sizes
relevant to thousands of basins and repeated parameter evaluations, within local
resource limits. Document durations, zones, candidates, parameter choices, fixed
subsets, and requested outputs. Smaller cases support correctness and scaling;
do not extrapolate unmeasured throughput as established results.

Report numerical fidelity, sustained throughput, and peak memory for the measured
workloads. Separate compilation/startup, data preparation, transfers, precomputation,
repeated evaluation, and output costs where relevant. Account for synchronization
or deferred device work. Explain when one-time work is amortized and how much
reuse the result assumes. Use enough repeated measurement to expose variability;
choose routine measurement mechanics during implementation.

Deliver runnable commands, synthetic generators/settings, comparison results,
and a concise report of hypotheses, observations, limitations, unsuccessful
approaches, and promising next steps. Keep useful experimental source and evidence
reproducible; no production cleanup or stable public API is required. Explain
trade-offs rather than declaring a universal winner. Return enough evidence for
the user to decide in #91 what to implement and whether further investigation is
needed. Link the report from both the experiment issue and #91.
