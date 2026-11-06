# HBV output storage and batch recovery

Save HBV simulation results for later analysis without accumulating full-run
histories in RAM or making execution excessively slow. Support comparing basins,
parameter sets and timestamps, including plotting discharge from several basins
over the same dates. Delivering this functionality completes
[discussion #128](https://github.com/CooperBigFoot/hydrologeez/issues/128);
publishing this vision alone does not.

## Recording and execution

- Let callers choose individual output variables: discharge, model states and
  internal fluxes. Do not limit selection to discharge-only versus all histories.
  Give variables discoverable names, units, dimensions and time semantics.
- Apply the chosen variables to every simulation in the run. No separate
  simulation or date selection per variable is required.
- Accept an optional recording start date and record through simulation end.
  Earlier days still execute normally and update memory, so warm-up can be omitted
  from storage without changing subsequent results.
- Keep native daily values and existing float64 precision. Frequency controls,
  resampling and aggregation are outside this work.
- Use bounded simulation batches and time windows where useful. Consume and write
  results without retaining all histories or expanding shared forcing across the
  entire parameter sweep. Exact batch/window sizing remains an engineering choice.

The motivating workload is the same library of 10,000 parameter sets across
CAMELS basins. Avoiding OOM is the priority, but an excessively slow implementation
is not useful. Scientific dates, basin eligibility and first-run variable choices
belong to the caller, not this storage layer.

## Resume completed work

Preserve completed batches when a sweep fails. Restarting must skip safely saved
completed batches and rerun the unfinished batch. Recovery inside an individual
simulation is not required. Time-window continuation for bounded memory does not
imply persistent recovery checkpoints at every window.

This explicitly supersedes the earlier exclusion of interrupted-run recovery in
#128 and the experiment: restarting one short simulation is acceptable; losing
thousands of completed parameter evaluations is not.

Do not mark a batch complete before its selected output is safely saved. Partial
writes must not appear as completed results or create duplicate results on resume.
Verify that a resumed run has matching inputs, parameters, identities and recording
settings; reject incompatible reuse rather than silently mixing runs. Preserve
existing outputs unless the caller explicitly chooses a separate destination or
an intentional replacement operation. Completion records and their physical
implementation remain engineering choices, not a new checkpoint service.

## Reading and identities

Reuse the CAMELS input approach where suitable: arrays stored on disk, explicit
identities, metadata-only opening and selection without loading the entire store.
The [CAMELS input vision](2026-11-04-camels-hbv-input-preparation-and-readers.md)
and current reader provide context, not a requirement to copy their file layout.

Readers must support selecting variables, basin IDs, parameter-set IDs and date
ranges for both time-series and cross-basin timestamp comparisons. Preserve
requested identity/order and return coordinates, units and dimensions with values,
not ambiguous unlabeled arrays. Use explicit date semantics; start-inclusive,
end-exclusive ranges are the discussed direction. Unknown or unavailable selections
must fail clearly rather than silently substitute or shorten results.

Discharge has basin, parameter-set and date identities. Zone and routing histories
also need their zone/lag associations, including unequal counts. State histories
are end-of-day values, not initial-state rows. Retain enough parameter information
to interpret and distinguish parameter sets. Expose recorded variables, available
identities/dates and completion status without loading histories. Incomplete runs
must not be presented as complete; exact incomplete-read behavior is left to the
implementer.

## Small agent-facing Python surface

Focus on functionality through a strongly typed Python API. The discussed surface
has three operations: run/resume a sweep, inspect a saved store, and read a selected
subset. Integrate with existing forcing, parameter and memory objects rather than
creating a second numerical model interface. Function names, argument grouping and
result types shown during discussion were illustrative, not fixed contracts.

A config-driven reproducible workflow may later sit on top of these operations.
Do not build it now. Likewise, no CLI, plotting API, query language, plugin system,
background job service or separate checkpoint API is required. Performance metrics,
clustering, calibration and scientific selection policy remain downstream.

## Evidence and implementation freedom

The [output experiment report](../../experiments/hbv-output-writing/README.md)
recommends bounded simulation batches plus time windows. It established exact
readback and continuation equivalence locally, but adopted no production format.
NPZ shards were an experimental sink, not a requirement. Exact format, physical
axis order, file grouping and completion bookkeeping remain open. Dense NumPy
storage is a useful candidate from the input work, not an output benchmark winner.
Do not choose compression by default solely to reduce bytes; the tested compressed
path was substantially slower.

Currently `hbv_light.model.simulate` returns discharge and complete final memory;
`history=True` allocates all state and flux histories for the call. Successive
start/stop calls can continue from final memory. Adapt this boundary as needed for
selective recording and bounded execution without changing hydrological equations.
Rewriting existing output execution is allowed; experimental scripts need not
become production dependencies.

Intended VM resources are 15 CPUs, 60 GB RAM and a reported 5 TB data disk, not
verified capacities. Local measurements do not prove VM safety: worker memory,
page cache, long-duration RSS and total output capacity remain relevant. Keep
concurrency deliberate and account for active results, writer buffers and shared
forcing. Full histories can exceed disk capacity even when RAM use is bounded.
Report limitations honestly; unavailable VM access must not be represented as
successful target validation.

## Evidence of completion

- Selected variables read back exactly match the unchanged model, including
  warm-up exclusion, final partial batches/windows and basin/parameter/date/zone/
  routing associations. Unselected variables are not stored.
- Reads support reordered multi-basin time series and timestamp comparisons with
  explicit coordinates; small selections and metadata access do not eagerly load
  full-run histories.
- Failure and resume tests establish that completed batches are skipped, unfinished
  work is safely recomputed, incompatible resumes are rejected, and the completed
  store matches an uninterrupted run without gaps or duplicates.
- A proportionate measurement reports end-to-end runtime, memory and stored size
  at representative sweep scale, with hardware, concurrency and measurement limits.
  Demonstrate bounded live output work rather than claim a hard RSS guarantee from
  array sizes alone. Reuse existing evidence; do not build a benchmark platform.
- Provide concise Python examples for variable selection, recording after warm-up,
  resume, and selected reads. Use project-native uv, typed Python, test-first
  development, Ruff and ty; run relevant tests and checks.
