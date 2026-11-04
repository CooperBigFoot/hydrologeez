# HBV input-path experiments

Develop an evidence-backed implementation recommendation for efficient data flow
from disk into HBV: **physical input layout, reader, and runtime representation
and consumption together**. Implement and run bounded local experiments, then
recommend the best complete path supported by the evidence. No format or existing
API is a predetermined winner.

This is the experiment handoff for [discussion #122](https://github.com/CooperBigFoot/hydrologeez/issues/122),
branched from [input-path discussion #90](https://github.com/CooperBigFoot/hydrologeez/issues/90)
within [Program #87](https://github.com/CooperBigFoot/hydrologeez/issues/87).
Returning the findings and recommendation completes the experimental work of #122,
not #90 or the Program. Publishing this vision does not complete any issue.
Experiments do not authorize production adoption; return the recommendation for
that decision.

## Purpose and boundaries

The research software must eventually enable the same library of roughly 10,000
parameter sets to run across roughly 400 basins over multiple years. That is context
for input reuse, **not a mandatory benchmark size**. CAMELS-US is the initial testing
dataset for this first iteration of the input system. Its lessons should inform
later datasets without building unused cross-model abstractions.

Inputs for a basin fit in RAM. Out-of-core processing and larger-than-RAM tests are
not requirements. Load and prepare inputs efficiently, then reuse them rather than
reload or duplicate forcing for each parameter set.

Runtime work includes changing array representation and how the numerical model
consumes it: indexing, memory access, loop structure, and avoidable copies or
allocations. The current interface is evidence, not a constraint to preserve at
any cost. **The science must not change:** preserve equations, process ordering,
and scientific behavior. This is not an audit of HBV's scientific validity.
Check implementation equivalence so a speedup cannot come from changed or omitted
calculations. Keep precision fixed initially to isolate layout effects; do not
introduce approximate physics or numerical methods as performance shortcuts.

Implementing parallelisation, scheduling, worker management, or simulation-grouping
strategies is outside this effort. Existing execution can measure actual input
consumption without redesigning parallel infrastructure. Also excluded: output
persistence, internal state/flux history optimization, scoring/calibration design,
zoning, GPU/backend redesign, cloud provisioning, and production adoption. Run the
model to produce streamflow with history disabled; output writing is not a metric
or design branch. Choose safe experiment sizes given the existing streamflow
allocation rather than implementing output infrastructure.

## Data and existing evidence

Use basin-average CAMELS Daymet precipitation and temperature, deriving daily mean
temperature and Oudin PET. No elevation/vegetation zones. Raw data is already at
`data/camels-us/raw/` in the canonical local checkout, which is Git-ignored. The
source files were verified against Zenodo sizes and MD5 checksums; source metadata
is retained there. Do not redownload or duplicate the raw archive into worktrees.
There is approximately 100 GB available for local work; avoid unnecessary copies
and large output materializations.

The archive contains 680 basin-mean Daymet files, whereas the attribute tables
contain 671 basin IDs. All attribute IDs have corresponding forcing filenames,
with nine extra forcing IDs. Match string IDs explicitly; do not infer the cohort
from directory counts. Extract/read only relevant members, not the much larger
collection of elevation-band forcing. Audit daily alignment, coverage, units, and
missing values before preparing equivalent fixtures. Do not silently fill missing
forcing with zeros. Choose documented valid windows for performance experiments;
return unresolved scientific missing-data choices rather than inventing a policy.
Make the temperature and Oudin preparation explicit and identical across candidates.
Observed flow is not model forcing, and scoring is out of scope.

The current `src/hydrologeez/hbv_light/model.py` accepts float64 forcing indexed as
`[day, packed basin zone, P/T/PET]`; one zone per basin suffices here. Simulation
basin indices already select shared forcing. Its per-simulation execution traverses
time, while conventional C-order forcing places basin values together within each
day. It does not require C-contiguity. Each `simulate` call also validates supplied
forcing again. These facts suggest locality and repeated-validation experiments,
not proven bottlenecks. Existing tests provide implementation-equivalence anchors.
Reinspect the current code rather than copying the superseded execution experiment.

The local `../hdx` precedent uses per-basin Parquet and a shared attribute table.
Its current readers primarily discover/validate metadata, not load scientific
values into HBV arrays. In particular, its scalar reader reads full file bytes
before selective decoding. Do not assume its metadata-oriented behavior is fast
selective input I/O or introduce an HDX dependency by default.

## Starting candidates

Compare complete paths, all providing the same prepared scientific values:

| Physical layout | Reader-to-model path | Main question |
| --- | --- | --- |
| Time-sorted Parquet per basin | Select basin files and dates/columns, decode, pack model inputs | Does simple basin selection outweigh file-opening and packing costs? |
| Consolidated Parquet sorted by basin/date, with basin-aligned row groups | Select/filter row groups, decode, pack model inputs | Can fewer files retain useful selection and reduce overhead? |
| Dense NumPy arrays with explicit basin/date metadata | Load selected arrays and adapt or directly expose them to the model | How much does avoiding table conversion help relative to uncompressed read volume and selection copies? |
| Compressed Zarr arrays | Read selected chunks into model-ready arrays | Does compression and array-oriented selection outweigh chunk/decompression overhead? |

Use reasonable initial configurations, not an exhaustive codec/chunk/layout
competition. Dense NumPy is a useful local baseline, not the chosen winner.
Memory mapping is optional as a loading variant, not an out-of-core requirement.
Treat chunk shape, row groups, compression and physical memory order as interacting
choices. A transpose view does not itself reorganize underlying bytes.

Across these candidates, compare physically time-major and basin-major memory
organization. Adapt numerical implementation when useful while preserving science.
Stored layout and in-memory layout need not be identical. Account for all conversion
and packing costs rather than making a fast kernel depend on hidden preparation.

## Experiment approach

Start with one independently checked canonical prepared CAMELS fixture and verify
that every reader returns matching identities, dates, values, units, and ordering.
Separate one-time source parsing/scientific preparation/store construction from
repeated loading. Holding prepared values fixed is the initial comparison baseline;
if preparation placement is varied, include its cost explicitly and preserve values.

Use proportionate combinations of few/many basins, contiguous/scattered selections,
and short/long date windows. Choose exact sizes as engineering settings, not user
research requirements. Increase scale only when needed to expose a meaningful cost.
Test prepared-input reuse across calls without reloading per candidate.

First isolate input representation with already-resident equivalent arrays; then
compare the complete disk-to-model paths. This is an iterative approach, not a
requirement to freeze runtime before studying readers. Measure selection, reading
and decoding, alignment/packing/conversion, validation, and actual HBV consumption.
Report total costs as well as stages, memory use, and stored size. Inspect strides
and materializations. Measure additional diagnostics such as file counts or bytes
read only where practical and meaningful.

Do not equate opening a memory map with consuming its data. Include first-touch
costs. Distinguish first-read observations from warm-cache repeats; restarting a
process does not clear the OS cache. Separate compilation from warmed execution,
record versions and relevant settings, and keep model outputs equivalent across
comparisons. Do not let hidden copies, deferred work, different thread settings, or
changed precision masquerade as storage improvements.

Refine only consequential uncertainties revealed by the initial measurements:
file overhead may warrant consolidation; excessive reads may warrant different
row groups/chunks; conversion costs may favor a direct representation; repeated
validation may justify a narrow reusable prepared-input boundary. Such a boundary
must not silently permit stale validation after mutable arrays change. Avoid a
framework or defensive API machinery without measured need. Execution restructuring
is allowed only within the agreed science-preserving input-consumption scope,
not as a parallelisation project.

Use project-native `uv`, typed Python, Ruff, ty, and test-first implementation.
Keep experiments and evidence distinct from the production API. A compact report
and reproducible commands are sufficient; no benchmark platform is required.

## Evidence and recommendation

Return a report with tested data/settings, correctness checks, reproducible commands,
measurements and variability, interpretation, limitations, and the recommended
physical layout + reader + runtime implementation together. Explain which costs
matter, which alternatives were rejected and why, and what remains uncertain.
Report construction cost separately from repeated use rather than assuming an
amortization period. Local disk evidence does not establish object-storage speed;
remote reads are not needed for these local experiments.

Success is a defensible recommendation and experimental implementation demonstrating
it—not a numerical latency target, a universal optimality claim, a full research
sweep, or production adoption. Findings return to #122 and inform #90. Preserve
useful evidence without retaining regenerable bulk outputs.

## Reference mechanics

These describe available mechanisms, not measured winners:

- [Arrow Dataset](https://arrow.apache.org/docs/python/dataset.html): projection, filtering, partitioning and discovery.
- [Parquet writer](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_table.html): row groups, statistics and compression.
- [NumPy loading](https://numpy.org/doc/stable/reference/generated/numpy.load.html) and [indexing](https://numpy.org/doc/stable/user/basics.indexing.html): mapping, views and selection copies.
- [Zarr performance](https://zarr.readthedocs.io/en/stable/user-guide/performance/): chunk shape and compression trade-offs.
- [Arrow to NumPy](https://arrow.apache.org/docs/python/generated/pyarrow.Array.html#pyarrow.Array.to_numpy): conversion constraints; zero-copy exposure of a decoded column does not make Parquet-to-model loading zero-copy.
