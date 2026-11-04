# HBV input-path experiments

This report and its two measurement files preserve the original UZL/K0 study.
See the [nonlinear-response rerun](nonlinear-response.md) for the current model;
use commit `ac7b5bc` to reproduce the historical results below.

Evidence for [#122](https://github.com/CooperBigFoot/hydrologeez/issues/122) and the [published vision](../../planning/visions/2026-11-04-hbv-input-path-experiments.md). This recommends an implementation for discussion #90; it does not adopt a production API.

## Recommendation

For repeated local HBV runs, use a dense float64 **basin × day × P/T/PET NumPy store**, explicit string basin IDs and contiguous Gregorian date metadata. Select and materialize only the requested inputs once, expose a transpose view as `[day, basin, variable]`, and reuse it across calls to the unchanged HBV implementation. The selection copy is real and measured: this is not a zero-copy loading claim. Both tested NumPy selections materialize basin-major bytes and avoid another copy when packed for basin-major runtime. The physical basin-major store recommendation rests on selection locality/read timing, not on eliminating a copy relative to the time-major store or a demonstrated major numerical-kernel speedup. C-order time-major remains reasonable when already resident; locality differences here are small and noisy.

Compressed block Zarr is a credible alternative when stored size matters. Do not add a reusable validation bypass now: validation is a small fraction of long-window consumption, and mutable arrays make stale validation a correctness risk. No numerical equations, process ordering, parallel execution, output infrastructure or production APIs were changed.

## Data and correctness

- Source: canonical `data/camels-us/raw/basin_timeseries_v1p2_metForcing_obsFlow.zip`, previously checksum-verified; archive not copied/extracted. Attribute membership comes from eight-character `camels_topo.txt` strings, never directory counts. Independent audit found all seven attribute tables agree on 671 IDs: 677 standard forcing filenames include six extras; three further archive names are irregular. They are not nine ordinary extra IDs.
- Sample: 128 evenly spaced indices in sorted attribute IDs, not random or spatially representative. IDs are in the JSON evidence. All selected complete source time series were checked for consecutive Gregorian Y/M/D, finite P/T, nonnegative P and temperature sentinels <=−999. No filling, no exclusions required. Prepared window is 1990-01-01 through 1999-12-31 (3652 days); leap days remain. Actual source hour12 is not converted to UTC. Native-grid Daymet's 365-record convention is not used.
- Exact header units/columns are checked. P is mm/day; T is `(Tmax+Tmin)/2` Celsius. PET reuses `scripts/daymet_prepare.py` Oudin Eq3 with FAO56 astronomy, lambda2.45 MJ/kg, solar constant0.0820 MJ/m²/min, denominator365, Gregorian DOY including366, PET zero when T<=−5. Latitude is **attribute gauge_lat**, consistently, not the differing forcing header latitude. This convention is not a claim of exact Morton astronomy. Independent anchors: equatorial equinox and FAO56 Example8 (latitude−20, DOY246, T20 → about3.286mm/day). Missing input is rejected before PET clipping.
- One canonical prepared float64 array supplies every store. Every measured reader checks exact values against it; Parquet additionally checks selected IDs/dates before packing. Synthetic tests request reordered string IDs and a partial date window. Runtime comparisons require bit-identical outlet **and complete final zone, groundwater, routing memory**. Existing independently worked HBV tests remain unchanged. A separate resident-order test covers freezing/melt temperatures and all final memory.

## Method and limits

Five cases: 4 basins×365 days; 128×365; first32×3652; every fourth of128×3652; all128×3652. Short windows start at the baseline's first day. Four simulation rows per basin share forcing (identical fixed parameters, not a calibration sample). Parameters: TT0, CFMAX2, CFR.05, CWH.1, SCF1, FC100, BETA1, LP.5; PERC1, UZL12, K0.2, K1.1, K2.05; MAXBAS2.5. Initial S/L=0, moisture50, groundwater/routing0. History disabled. Largest outlet is3652×512 float64 (~14.3MiB), not 400×10000 simulations.

Numba and Arrow use one thread; final Zarr async concurrency, thread pool and codec pipeline worker limit are1. Initial Zarr used its defaults (async10), so initial-to-final Zarr timings are diagnostic, not a controlled chunk-only comparison. No worker/scheduling implementation was added. Two Numba signatures are warmed before measurement, with compilation reported separately. Resident experiments reuse the same values for four calls with fresh identical memory, without reloading.

Every disk path has one first observed read and three warm repeats, in fixed order, on local macOS/APFS. Stores were just written: **no cold-cache claim**, no OS cache eviction, randomization or remote/object-storage inference. mmap opening is not consumption: advanced indexing materializes selected data and an explicit sum touches all selected values inside read/decode timing. This includes memory-mapped page faults but does not measure physical device bytes. Small mmap calls show several-ms overhead here; no universal disk-latency inference is justified.

Stages are metadata/selection; read/decode/first touch; alignment/packing; and `simulate` including its normal validation. A separate validation probe is diagnostic and **not added again to totals**. Input object/parameter construction outside simulate is not included in totals, identically for all paths. Output equality checks are outside timers. Parquet alignment checks/sort are included in packing. NumPy/Zarr identity comes from their shared explicit metadata. No output writing is timed. Peak process RSS includes fixture, stores, JIT and temporary outputs; it is not isolated per-reader memory. Raw records include selected input bytes/strides; no precise peak allocation attribution is claimed.

Parquet uses zstd level3, 365-row groups that never cross a basin, time-sorted basin files or one consolidated file. Both project the same columns and filter dates/IDs through Arrow Dataset with threads disabled. Final packing sorts once and reshapes, rather than filtering the complete table for every basin. NumPy stores C-order time-major and C-order basin-major arrays, read via mmap. Zarr v3 uses Blosc zstd3 shuffle with chunks `(365,16,3)` time-major or `(16,365,3)` basin-major. Initial basin chunks `(1,365,3)` caused excessive chunk work and were refined. Sparse selection can overread block chunks. Both runtime representations are tested for every physical candidate: C `[day,basin,3]` strides `(24*basins,24,8)`, or basin-major bytes exposed through a view, strides `(24,24*days,8)`. A transpose alone does not reorganize bytes; `ascontiguousarray` packing and its cost are explicit.

## Measurements

All times below are milliseconds. Warm results are medians of three repeats; the evidence retains every stage, first observation and warm repeat. The following complete-path table uses basin-major runtime for a consistent comparison. Results are local observations, not universal winners.

| Physical path | few-short | many-short | contiguous-long | scattered-long | many-long |
|---|---:|---:|---:|---:|---:|
| numpy-time | 5.80 | 26.28 | 52.51 | 55.97 | 200.76 |
| numpy-basin | 6.06 | 26.22 | 52.54 | 51.85 | 198.97 |
| parquet-files | 1.88 | 43.37 | 78.85 | 78.73 | 314.57 |
| parquet-consolidated | 16.81 | 56.76 | 93.91 | 94.20 | 333.31 |
| zarr-time | 2.23 | 27.00 | 60.60 | 91.21 | 242.02 |
| zarr-basin | 2.05 | 26.25 | 63.03 | 95.00 | 236.13 |

Many-long detail, basin-major runtime (warm median; total range across three repeats):

| Path | Load | Simulate | Total range | First observed total |
|---|---:|---:|---:|---:|
| numpy-time | 10.12 | 190.64 | 198.96–200.77 | 201.00 |
| numpy-basin | 6.89 | 191.76 | 197.23–202.12 | 193.11 |
| parquet-files | 124.47 | 191.85 | 314.05–317.64 | 316.89 |
| parquet-consolidated | 144.10 | 189.21 | 330.27–341.16 | 333.08 |
| zarr-time | 50.00 | 192.28 | 241.76–244.03 | 248.75 |
| zarr-basin | 46.09 | 189.97 | 231.91–238.30 | 235.86 |

Resident consumption, warm median [min–max], milliseconds:

| Case | Time-major | Basin-major |
|---|---:|---:|
| few-short | 0.78 [0.76–0.80] | 0.81 [0.79–0.90] |
| many-short | 21.28 [20.88–21.37] | 21.13 [20.95–21.18] |
| contiguous-long | 47.55 [47.41–49.22] | 48.02 [47.97–51.25] |
| scattered-long | 47.77 [47.41–48.01] | 46.85 [46.69–47.34] |
| many-long | 208.57 [203.74–208.75] | 190.13 [189.88–191.40] |

Construction is separate, without assumed amortization. Source parsing/preparation 2.065s; JIT warm-up 0.066s (includes possible Numba cache loads); peak process RSS 742.6MiB on macOS.

| Store | Construction ms | Stored MiB | Files |
|---|---:|---:|---:|
| numpy-time | 1.16 | 10.70 | 1 |
| numpy-basin | 1.06 | 10.70 | 1 |
| parquet-files | 261.06 | 9.31 | 128 |
| parquet-consolidated | 213.87 | 9.25 | 1 |
| zarr-time | 175.85 | 7.56 | 89 |
| zarr-basin | 170.24 | 7.59 | 89 |

Shared metadata is 1620 bytes. Add 8.00ms table preparation for either Parquet construction; basin-major array packing adds 2.60ms. Values occupy 10.70MiB. A selected copy occupies `days*basins*3*8` bytes; converting physical order may briefly need a second such array. Parquet also holds decoded columns and sort buffers; Zarr holds chunks plus selection. No fsync/durability timing.

## Interpretation and remaining uncertainty

Dense arrays avoid table reconstruction and are strongest for the larger repeated-local cases. Simple per-basin Parquet is competitive for a few short series, but consolidation did not overcome discovery/filter/packing costs here. Compression saves storage but is not free; block Zarr is materially better than the initial one-basin chunks, yet the final single-worker path is slower than dense arrays on long windows. This does not prove Parquet or Zarr intrinsically slower under other readers/chunks/storage systems.

Initial measurements motivated only three refinements: replace thousands of repeated JSON date labels with checked daily start/count metadata; sort/reshape decoded Parquet once; and enlarge basin-major Zarr chunks from1 to16 basins. Initial evidence uses the first commit's implementation and default Zarr concurrency, whereas final evidence fixes all worker limits to1. Numerical consumption is unchanged. Validation stays enabled: long-window probes are roughly1ms at128 basins, small relative to ~190ms simulation. A new prepared-input boundary would add more correctness responsibility than measured benefit.

No resident speedup warrants a new numerical implementation. The existing per-simulation time loop already consumes physically basin-major forcing without a production change. Four repeated resident calls demonstrate reuse, not a universal parameter-count scaling law. Higher reuse amortizes loading, making the small runtime differences more relevant; more parameter counts/hardware and randomized repetitions would be needed before choosing on those differences alone. Memory measurements are process-wide, cache state uncontrolled, and three warm repeats are modest. No claims cover missing-data scientific policy, all671 basins, other datasets, remote storage, parallel strategies, precision changes, or numerical validity.

## Reproduce and verify

From the implementation checkout (replace RAW with the canonical absolute path; do not copy the archive):

```sh
uv sync --group experiments --group acquisition
uv run --group experiments python -m scripts.hbv_input_path --raw RAW --output .hbv-input-data --evidence experiments/hbv-input-path/measurements.json
uv run --group experiments --group acquisition pytest -q
uv run --group experiments ruff check .
uv run --group experiments ty check
```

The experiments group adds Arrow, Zarr and netCDF4 (the last is an incidental import dependency of the reused Oudin module). Versions/platform are recorded in JSON. Generated stores are ignored; only compact measured evidence is tracked. `initial-measurements.json` corresponds to commit `b8c859f` before refinements; `measurements.json` is the final script. Raw JSON is compact to avoid thousands of low-value diff lines; each record contains timings, strides and repeat identity. Full verification:54 tests pass; Ruff and ty pass. Necessity review kept one ~350-line experimental script,16 focused tests, two small evidence files and this report; no benchmark framework, dependency in the production group, copied archive or product API was added.
