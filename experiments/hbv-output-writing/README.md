# HBV output writing: local screening

Evidence for [#131](https://github.com/CooperBigFoot/hydrologeez/issues/131), returning to
[#128](https://github.com/CooperBigFoot/hydrologeez/issues/128), under the
[published vision](../../planning/visions/2026-11-04-hbv-output-writing-experiments.md).
No production API or format is adopted.

## Recommendation

Use **simulation batches plus time windows**, consuming each result before the next
call, as the next implementation direction. In this screening, uncompressed NPZ
shards were a simple effective output sink. At 2,048 simulations × 1,460 days with
all histories, tiles took 0.555 s versus the resident baseline's 0.717 s, with median
process peak 290 versus 698 MiB. At 2,048 × 365, tiles were 19% slower (0.160 versus
0.134 s). That measured overhead is a reasonable exchange for avoiding full-run
history allocation; it is not a universal acceptable-runtime threshold.

Simulation batching alone still grows with the complete time axis. Compression
reduced the largest comparable output from 487 to 230 MiB but cost 12× tiled
runtime and higher process peaks. Do not choose it by default from these results.
Direct mapped output was not tested: it would require replacing the model's
allocation path, and mapped resident pages would still need measurement. These
four execution/storage combinations are not an exhaustive format comparison.

## Workload and mechanics

- Real CAMELS Daymet lumped forcing, basin IDs 01013500, 03241500, 08066200,
  14400000: four evenly spaced rows of sorted CAMELS topographic attributes.
  Dates begin 1990-01-01; durations are 365 and 1,460 actual daily intervals.
  P/T/Oudin PET preparation reuses the existing input study; raw archives are read
  in place, not copied. These basins are a screening sample, not a scientific design.
- Each basin receives the **same distinct parameter library**, not identical copies.
  For parameter ID j, x=j/(N−1). Zone parameters TT, CFMAX, CFR, CWH, SCF, FC,
  BETA, LP are −0.5+x, 1.5+x, 0.02+0.06x, 0.05+0.1x, 0.8+0.4x, 80+120x,
  1+x, 0.4+0.4x. PERC, Alpha, K1, K2 are 0.5+x, 0.2+0.6x, 0.05+0.1x,
  0.02+0.06x; MAXBAS=1+3.5x. This correlated engineering sweep is not calibration
  or a search-space sample. Soil starts at FC/2; other memory starts at zero.
- No hydrological equations or production source changed. All retained arrays are
  float64. Q-only retains outlet and final memory; full retains outlet, zone,
  groundwater and routing histories, all zone/basin fluxes, and final memory.
- One process, one Numba thread, no concurrent writers. `resident` computes all
  rows/days before saving; `batch` uses 256 rows and all days; `tile` uses 256 rows
  ×365 days; `compressed` uses the same tiles with NumPy ZIP deflate defaults.
  Uncompressed candidates use ZIP_STORED NPZ, not raw NPY or mmap.
- Each shard carries basin indices, parameter IDs and actual parameter arrays,
  local zone/routing offsets, final memory and outputs. The manifest maps shard
  row/time ranges, ordered basin IDs and Gregorian ordinal dates. Row r maps to
  basin r%4 and parameter r//4; packed zones retain forcing order within each basin;
  routing lags retain next-arrival order. State rows are end-of-day, not initial state.
  Final memory is saved at every window boundary for equivalent continuation.

## Correctness, separate from timing

Eight tests compare every retained value read back from every candidate with an
unchanged monolithic HBV call, using exact array equality. They cover Q-only/full,
unequal basin zone counts (1 and 2), unequal routing lengths including zero,
partial final batches/windows, nontrivial snow/soil forcing, parameter arrays,
independently expected basin/parameter IDs, date/packed offsets, and complete final
memory. A coverage array checks every day/simulation occurs exactly once. These
small deterministic tests establish execution/storage equivalence, not independent
physical validation of HBV. Large performance cases do not retain a second full
reference or read back their entire output during measurement.

The new test first failed because the experimental module did not exist. Final
validation: 88 tests pass; Ruff check/format and ty pass. No existing tests removed.

## Measurement scope

[Raw records](measurements.json): three independent fresh processes per combination,
run sequentially in resident/batch/tile/compressed order within each repeat. Case
order was fixed, not randomized. Hardware: Apple M4 Max, 16 logical CPUs, 64 GiB
RAM, macOS 15.7.3; local filesystem, device/cache state not characterized. Python
3.13.14, NumPy 2.5.4, Numba 0.68.1. No authorized VM connection was available;
these are **not** measurements of the proposed 15-CPU/60-GB VM or its 5-TB disk.

Simulation/write timers include parameter and initial-memory generation, model
validation, allocation, simulation, serialization, shard metadata, file flush and
`fsync` for every shard and the manifest. Timers end after the manifest closes.
Input preparation (archive read, decode, PET) and Numba warm-up are reported
separately in each record; total excluding warm-up adds preparation to write time.
Warm-up may use the JIT cache. Directory entries are not fsynced; macOS `fsync`
is not a guarantee that a device's volatile cache has drained (no F_FULLFSYNC).
These are application completion times, **not physical device throughput or crash
recovery guarantees**. No cache eviction or disk saturation was attempted.

Peak RSS is the fresh process lifetime high-water mark, including imports, input
preparation and JIT, not an allocation delta or system-wide memory. Preparation
and warm-up floors can mask small cases. Ordinary buffered file writes can consume
OS page cache outside process RSS. No candidate uses mmap; no measurement of RSS
here proves the disk/page cache will stay below a VM-wide limit.

## Results

Median (minimum–maximum), three processes; stored size is logical file bytes.
Q=outlet plus final memory, full=all histories plus outlet/final memory.

| Simulations × days | Saved | Path | Simulation + write s | Peak RSS MiB | Stored MiB |
| --- | --- | --- | --- | --- | --- |
| 512 × 365 | Q | resident | 0.023 (0.022–0.023) | 185 (181–198) | 1.5 |
| 512 × 365 | Q | batch | 0.023 (0.022–0.024) | 183 (183–187) | 1.5 |
| 512 × 365 | Q | tile | 0.023 (0.022–0.023) | 179 (178–182) | 1.5 |
| 512 × 365 | Q | compressed | 0.054 (0.053–0.054) | 180 (179–182) | 1.4 |
| 512 × 365 | full | resident | 0.042 (0.040–0.042) | 231 (228–243) | 30.4 |
| 512 × 365 | full | batch | 0.044 (0.044–0.057) | 226 (221–227) | 30.4 |
| 512 × 365 | full | tile | 0.042 (0.041–0.048) | 225 (211–226) | 30.4 |
| 512 × 365 | full | compressed | 0.411 (0.408–0.414) | 227 (227–232) | 14.5 |
| 512 × 1460 | full | resident | 0.173 (0.154–0.191) | 332 (321–333) | 121.5 |
| 512 × 1460 | full | batch | 0.172 (0.170–0.184) | 319 (308–325) | 121.5 |
| 512 × 1460 | full | tile | 0.139 (0.136–0.152) | 240 (235–242) | 121.8 |
| 512 × 1460 | full | compressed | 1.683 (1.671–1.722) | 324 (268–345) | 58.1 |
| 2048 × 365 | full | resident | 0.134 (0.134–0.135) | 320 (320–322) | 121.8 |
| 2048 × 365 | full | batch | 0.144 (0.142–0.145) | 255 (254–266) | 121.8 |
| 2048 × 365 | full | tile | 0.160 (0.155–0.160) | 262 (261–269) | 121.8 |
| 2048 × 365 | full | compressed | 1.703 (1.665–1.746) | 281 (276–292) | 57.6 |
| 2048 × 1460 | full | resident | 0.717 (0.717–0.784) | 698 (692–701) | 485.9 |
| 2048 × 1460 | full | batch | 0.678 (0.663–0.736) | 385 (385–422) | 485.9 |
| 2048 × 1460 | full | tile | 0.555 (0.541–0.647) | 290 (277–313) | 487.1 |
| 2048 × 1460 | full | compressed | 6.712 (6.675–6.733) | 445 (429–447) | 230.5 |

The largest comparable full-history output has 1/8/32/32 shards for
resident/batch/tile/compressed respectively, plus one manifest. Repeated parameters
and final memory explain the small tiled storage premium. Full histories increased
512×365 output from 1.5 to 30.4 MiB and resident runtime from 0.023 to 0.042 s.

### Larger bounded-work run

The same tiled path also ran **10,000 distinct parameter sets across four basins**:
40,000 simulations ×1,460 days, full histories, three fresh processes. Each wrote
9,978,407,847 bytes (9.29 GiB) in 628 shards plus one manifest. Runtime median
11.14 s
(range 10.81–11.21);
peak RSS median 535 MiB
(range 525–567).
Preparation took 0.110–3.790 s across those processes; total excluding warm-up
was 10.926–15.001 s. The first run was not a controlled cold-cache measurement.
No resident baseline was run at that scale. This is a 10,000-set library test, not
an all-CAMELS-basin run or a multi-decade run.

## Memory ownership and remaining limits

Only one tile's result histories, active parameter batch, previous/final memory,
and shared forcing are live. References to the old result and serialization
mapping are deleted before computing the next tile. No full-run simulation array
or full parameter matrix is accumulated. Largest explicit saved-array payload is
17,246,224 bytes in the 40,000-row case; model memory copies and temporary daily
flux/validation/routing arrays add to that. NumPy ZIP writing has bounded chunk
buffers; compression also owns deflate buffers. Python/NumPy allocator retention
and JIT are included in RSS but not that payload count.

Peak RSS is **not flat**: 225 MiB at 512×365 grows to 290 MiB at 2,048×1,460 and
roughly 535 MiB at 40,000×1,460.
The bounded live output arrays therefore must not be described as a measured hard
RSS bound. Allocator retention, preparation and OS accounting were not isolated.
The gap between ~9.3 GiB written and process peak supports avoiding full-run
accumulation on this machine, not an unconditional OOM guarantee.

Shared forcing grows with days×basin zones; this experiment leaves the input
boundary unchanged. Dates and the small manifest shard list also grow with run
size. These are not full histories, but strict constant-memory metadata is not
claimed. Scaling workers would multiply active tile, forcing/JIT and writer costs;
15 concurrent writers were not measured. Before a scientific full-scale run,
validate on the actual mounted VM disk, measure aggregate process plus page-cache
pressure under its intended concurrency, long-duration RSS behavior, space/inodes,
and end-to-end runtime. A full all-basin long-history output may exceed 5 TB even
when process memory is safe; do not infer storage capacity from these four basins.

## Reproduce

From the repository, with canonical CAMELS raw inputs available:

```sh
uv sync --group experiments --group acquisition
uv run --group experiments --group acquisition pytest tests/test_hbv_output_writing.py
uv run --group experiments --group acquisition python -m scripts.hbv_output_writing --raw /absolute/canonical/data/camels-us/raw --output data/output-tile --evidence data/output-tile.json --kind tile --days 1460 --copies 512 --history
```

Use a new output directory per process (existing paths are rejected). For the
comparison, run each of resident, batch, tile, compressed three times for
(days,copies,history) = (365,128,false), (365,128,true), (1460,128,true),
(365,512,true), (1460,512,true). Omit `--history` for false. Then run tile with
`--days 1460 --copies 10000 --history` three times. Do not run measurements
concurrently. Remove generated stores between cases after collecting the JSON
record; stores are disposable, not evidence. The original run used the canonical
checkout's identical project environment while this worktree's dependencies synced;
final tests and the large runs used this worktree's `uv` environment.
