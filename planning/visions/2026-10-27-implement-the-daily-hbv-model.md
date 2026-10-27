# Implement the daily HBV model

Deliver the HBV model in the `hydrologeez` package, following
[the mathematical description](../../docs/hbv-light.md) and using the completed
execution experiments to guide implementation. This is the confirmed implementation
scope of [discussion #91](https://github.com/CooperBigFoot/hydrologeez/issues/91).
Delivery and verification of this implementation, including the cleanup below,
complete #91. Publishing this vision leaves the issue open and starts no implementation.

## Model and execution

Implement the description's ordered daily rules, not a claim of verified equivalence
to an HBV-light executable. Preserve snow partition, melt/refreezing and retention,
soil recharge and evaporation, groundwater allocation, integrated fractional
triangular routing, and their specified timing and available-water limits.

Snow and soil memory belongs to zones. Area-weighted zone recharge feeds basin-wide
groundwater and routing. Support single-zone and multi-zone basins and many
independent basin–parameter simulations. Parameter sharing never merges zone states.
Preserve the per-parameter sharing meanings agreed in the
[configuration discussion](2026-10-27-bucket-model-configuration-surface.md) through
supplied resolved values; no configuration machinery is required.

Use NumPy arrays and Numba-compiled CPU stepping sequentially through time, parallel
across independent simulations, with shared basin forcings rather than candidate
copies. Float64 is the numerical baseline. The CPU experiment provides an implementation
starting point, not a requirement to preserve its experimental interfaces or fixed padding.
Do not carry over synthetic restrictions on zone counts, routing durations, parameter
ranges, or initial conditions as model restrictions. Exact Python interfaces, source
organization, batching, and internal layout are implementation choices.

The evidence is [PR #101](https://github.com/CooperBigFoot/hydrologeez/pull/101).
On the local M4 Max, eight-thread CPU execution beat the tested MLX GPU approach;
forcing expansion increased memory without a clear speed benefit. The isolated
temporal probe was slower when successful and frequently failed. These findings
support building the CPU model now, not further backend experiments or a universal
performance claim. No new benchmark project or fixed throughput target is required.

## Inputs and results

Callers supply prepared daily zone precipitation, temperature and PET, zone area
fractions, resolved parameter values, and explicit initial memory. Preserve the
model's units, domains and interval alignment. Missing values or scientific defaults
must not be invented. Geographic preparation, elevation correction of raw station
inputs and PET estimation belong upstream of these prepared inputs.

Return daily outlet predictions and complete final memory. When requested, expose
full timestep state histories, including routing memory, and model fluxes/quantities.
Preserve basin, parameter-set, zone and timestep associations and distinguish initial
memory from end-of-day states. Prediction-only execution must not retain unrequested
full histories. Keep bounded-memory history retrieval possible without requiring all
histories in RAM or adopting a permanent disk format or storage subsystem.

Restart from complete supplied memory with identical parameters and future forcings
must preserve continuation. Warm-up is ordinary evolution with the candidate's own
parameters; it must not reuse memory produced by different parameters. Calendar-based
warm-up configuration, scoring and assessment orchestration are outside this model.

## Verification and removal of experiments

Use focused model tests to establish agreement with the description: its worked day,
meaningful threshold and available-water cases, area-weighted zone composition,
fractional routing, full-memory restart, output timing, and water accounting including
water in transit. Check requested histories as well as discharge. Agreement between
two implementations alone is not independent evidence of the daily rules.

Use the repository's test-first guidance and project-native `uv`, Ruff and ty checks.
Retain useful independent expectations from experimental tests where appropriate,
but make the delivered model's tests independent of the experimental tree.

Once the package model is implemented and verified, delete the entire HBV experimental
tree introduced by #101, including experimental source, reports and raw measurements,
and remove its experiment-specific tests. Retain focused tests for the delivered
model, not parallel experimental implementations. Remove experiment-only dependencies
and settings that are no longer used; retain anything needed by the actual model.
Git history and #101 preserve the research evidence. Existing unrelated files and
tests are not cleanup targets.

## Explicit exclusions

The deliverable is the model implementation and its focused tests. No separate runnable
example or additional documentation deliverables are requested. The existing model
description remains the behavioral reference. Do not implement configuration parsing,
parameter-table resolution, sampling, calibration, data readers, geographic preparation,
a CLI, or experiment orchestration. The configuration vision supplies usage context,
not an instruction to implement that larger surface here. GPU and temporal execution
research are complete for this decision and are not part of this delivery.
