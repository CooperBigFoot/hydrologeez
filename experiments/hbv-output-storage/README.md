# HBV output storage measurements

Production implementation of the [published vision](../../planning/visions/2026-11-05-hbv-output-storage-and-batch-recovery.md).
[API and examples](../../docs/hbv-output.md). This is a local engineering check,
not VM capacity validation or a new benchmark platform.

## Method

Reuse the [prior output experiment](../hbv-output-writing/README.md)'s real CAMELS
Daymet forcing, four basins, distinct 10,000-set correlated parameter library,
initial memory, 256-row batches and 365-day windows. The production path saves all
19 individual variables at float64 precision; unlike the old experiment it does
not duplicate discharge in basin fluxes or persist final-memory checkpoints.
Inputs are prepared in place from canonical raw archives; generated stores are
disposable and were removed after collecting JSON evidence.

One fresh process, one Numba thread, one writer per case. Hardware: Apple M4 Max,
16 logical CPUs, 64 GiB RAM, macOS 15.7.3, local filesystem; device/cache state not
characterized. Library versions and exact timings are in the linked JSON files.
Timers include parameter/initial-memory construction, full input hashing,
validation, simulation, file and directory fsync, atomic batch publication and
final metadata inspection. Input preparation and JIT warm-up are separate;
end-to-end excluding warm-up adds preparation. Peak RSS is process lifetime,
including imports/preparation/JIT and one post-timing tile used to measure live
history bytes. No cache eviction, randomization or sustained multi-day run.

## Layout decision

The first implementation used a file per variable/window. The focused follow-up
uses a file per variable/batch: NumPy writes its NPY header, then each tile appends
in date order through a bounded streaming writer. No full-history buffer or writer
mmap is used. One contiguous variable-tile buffer avoids element-by-element
writing of strided model columns and is released before the next variable. Readers still mmap selected date/column slices. Fsync occurs after
all windows of the batch are written, before atomic batch publication. Recovery
still reruns the whole unfinished batch.

The old layout's [sequential large run](large-sequential.json) took 44.74 s and
12,562 files. The [first large run](large.json) took 50.35 s but overlapped tests,
so it is retained as confounded evidence, not the comparison. Small initial-layout
records: [365 days](small.json), [1,460 days](longsmall.json). The initial layout
source is not retained as another production backend; measurements motivated its
replacement, not a format tournament. Per-batch files alone reduced the large run
only to [43.18 s](batch-files-large.json). Writing each strided array with `tofile`
was slow. A contiguous single-variable tile buffer reduced it to 11.41 s with the
same stored bytes/files. [Intermediate small case](batch-files-small.json).
The final script reproduces only the final layout. The final large run was started
before the small case; timing overlap with that <1-second case cannot be ruled out
from these records, so the large result is a screening observation, not an isolated
throughput benchmark.

## Final layout results

| Simulations × days | Run s | End-to-end excluding JIT s | Peak RSS MiB | Stored GiB | Files |
| --- | ---: | ---: | ---: | ---: | ---: |
| [2,048 ×365](buffered-small.json) | 0.200 | 0.311 | 268.5 | 0.113 | 186 |
| [2,048 ×1,460](buffered-longsmall.json) | 0.730 | 0.867 | 344.6 | 0.452 | 186 |
| [40,000 ×1,460](buffered-large.json) | 11.407 | 11.515 | 417.6 | 8.832 | 3,613 |

One observation per case, not statistical bounds. Prior experimental NPZ tiles
at the large scale took median 11.14 s and 535 MiB over three runs. These are not
paired timings: the production path adds fingerprints, single-variable files,
durable directory commits and stronger inspection. Do not infer disk throughput
or a universal acceptable slowdown from this local comparison.

Every final case has the same largest live history tile: 17,192,960 bytes
(~16.4 MiB), including redundant model outlet/basin-flux discharge but excluding
final-memory copies, parameters and daily temporaries. The largest variable writer
buffer in this one-zone case is routing: 365×256×4×8 = 2,990,080 bytes, bringing
explicit live output history+writer payload to at most 20,183,040 bytes (~19.2 MiB). Each old result is released
before the next window. Inputs/parameters/metadata still scale with workload;
RSS need not be flat. The gap between full output size and measured RSS supports
bounded live output work, not a hard RSS/OOM guarantee. Writer buffers and OS page
cache remain additional memory; long-duration retention and aggregate worker
memory are unmeasured.

At 671 basins ×10,000 parameter sets, batch=256 gives 26,211 batches. All 19
variables plus three parameter arrays and one marker produce about 603,000 files,
independent of duration. Batch size is tunable before a run but is part of resume
identity. Increasing it trades fewer files for more live tile memory. Output bytes
still grow with duration, zones and routing lags and can exceed the reported 5 TB
VM disk. Validate actual VM disk space/inodes, page-cache pressure, intended Numba
threads, long-duration RSS and elapsed runtime before scientific full-scale use.
No authorized VM connection was available; none of these results proves safety on
the proposed 15-CPU/60-GB VM.

## Reproduce

From the repository with raw CAMELS available, choose a new store per process:

```sh
uv sync --group experiments --group acquisition
uv run --group experiments --group acquisition python -m scripts.measure_hbv_output --raw /absolute/data/camels-us/raw --output data/output-measure --evidence measurement.json --copies 10000 --days 1460
```

For smaller fixed-window cases use `--copies 512 --days 365` and `--copies 512
--days 1460`. Remove generated stores after retaining JSON, not before it is read.

## Correctness evidence

Tests compare every selected variable exactly with unchanged monolithic HBV,
including warm-up exclusion, reordered basin/parameter pairs, final partial
batches/windows, unequal zones and zero/unequal routing counts. They establish
selected mmap reads and metadata-only opening, missing-selection rejection,
partial-window and pre-commit failures, completed-batch preservation, incompatible
input/settings rejection and missing committed-file detection. No model equations
or existing coverage were changed. Validation commands/results are recorded in
[validation.txt](validation.txt).
