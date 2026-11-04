# HBV output writing experiments

Compare possible implementations for saving HBV discharge, model-state histories,
and internal-flux histories without exhausting RAM or making runs excessively slow.
Return evidence and an implementation recommendation to [#128](https://github.com/CooperBigFoot/hydrologeez/issues/128).
This is experimental implementation work for [#131](https://github.com/CooperBigFoot/hydrologeez/issues/131):
completing the comparison and reporting its findings completes #131, not #128.
Publishing this vision completes neither issue and does not adopt a production API.

## Priorities and context

Avoiding out-of-memory failures is the primary requirement. Runtime also matters:
a memory-efficient but extremely slow implementation is not a useful solution.
There is no agreed absolute runtime threshold; show the trade-offs against a
comparable baseline rather than inventing one. Analysis happens after runs, so
live analysis and low-latency output availability are not requirements.

The motivating scale is the same library of 10,000 parameter sets across CAMELS
basins. Intended VM resources are 15 CPUs and 60 GB RAM, with a reported attached
5 TB virtual data disk that may still need mounting. These are planning facts,
not verified machine measurements. Do not publish private connection details.

The experiment is not a design of the scientific run or the input boundary.
Choose representative workloads and measurement mechanics as engineering choices;
do not require the user to specify basin selection, scientific dates, or which
histories the first scientific run will enable. Include discharge, states, and
internal fluxes in the comparison's coverage.

## Starting evidence and freedom to explore

`src/hydrologeez/hbv_light/model.py` currently allocates discharge for every call.
With history enabled it also allocates complete state and internal-flux histories.
Arrays use float64; state includes zone, groundwater, and routing memory. Simulation
rows associate basins with parameter sets, and packed zone/routing axes must retain
their associations when stored. Successive time-window calls can continue from
complete final memory, but this is only one candidate approach.

Compare a proportionate set of useful alternatives. Simulation batching, time-window
continuation, and direct disk-backed output are candidates, not mandatory designs
or a predetermined winner. Storage layout, format, buffering, and execution strategy
remain open. Rewriting experimental execution/output code is allowed; changing the
hydrological equations is not part of this work.

The [input-path study](../../experiments/hbv-input-path/README.md) and its linked
current-model rerun provide context, not output-writing evidence. They used small
ensembles and did not measure output writing at the motivating scale. Do not infer
an output format winner from their input-read results.

## Evidence to deliver

Use project-native tools and a small experimental implementation, not an exhaustive
format tournament or a reusable benchmark platform. Choose workloads that expose
memory growth and runtime trade-offs, including the cost of recording full histories.
Use real HBV behavior and document data, parameter diversity, dimensions, precision,
thread/worker settings, and hardware so conclusions have a clear scope.

Establish correctness separately from performance. Compare equivalent retained
values, final memory, and basin/parameter/date/zone/routing associations; verify that
written results can be read back correctly. Precision reduction is not a settled
requirement and must not silently make comparisons inequivalent.

Measure complete simulation-and-write runtime, peak memory, and stored size.
Explain memory ownership across concurrent work, simulation results, writer buffers,
and temporary copies. In particular, distinguish process memory from operating-system
page cache for disk-backed approaches; an allocation-free-looking interface alone
is not proof of bounded memory. Account for preparation and deferred write work,
and state when timings stop rather than claiming device throughput from buffered
writes. Show how memory behaves as total run size increases under bounded work units.

Use a comparable baseline where it safely fits. Report repeats or other evidence of
variability, numerical correctness, and measurement limitations. Local screening is
useful, but local results are not verified VM performance. If VM access is unavailable,
report that limit and any remaining validation need rather than blocking all progress
or implying target-machine validation occurred.

Deliver concise reproducible evidence, a recommendation, and its runtime/memory/storage
trade-offs to #128. Explain why the recommended approach avoids accumulating the full
run in RAM and why its slowdown is acceptable relative to measured alternatives.
Clearly identify unresolved evidence rather than presenting extrapolation as a result.

## Boundaries

No production API adoption or full production implementation in this effort. No input
boundary redesign, scientific-run design, performance metrics, basin-by-parameter
performance matrix, clustering, or interrupted-run recovery. Restarting interrupted
runs is acceptable. Storage formats and execution mechanisms are choices for this
comparison, not requirements inherited from earlier proposals.
