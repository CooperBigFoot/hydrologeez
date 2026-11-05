# Seeded calibration experiments

Supports the [vision](../visions/2026-11-05-seeded-ensemble-calibration-for-hbv.md).

## Decisions

Initially exercised ctrl-freak 0.2.1 batch callbacks. The user then requested
local ownership: final code owns minimal GA/NSGA-II and bounded operators under
`calibration`, with MIT attribution. No framework, worker pools or dependency
remains. Existing HBV runtime is unchanged.

Full-period candidate batches 8–64 reduced setup overhead versus one candidate.
Keep configurable batch size 64; 32 was essentially tied. Exact period scoring
is retained rather than averaging windows. Metrics dominate warm simulation in
this small workload; observation-rank precomputation remains a possible measured
improvement, not implemented as a second scoring path yet.

## Measurements, 2026-11-05

Final local optimizer, Apple M4 Max, macOS 15.7.3 arm64, Python 3.13.14,
NumPy 2.5.4, Numba 0.68.1, `NUMBA_NUM_THREADS=2`. Prepared metadata has 671
basins. Tested only 01013500 over 1980-01-01–2015-01-01 (12,784 days), training
1981-10-01–2000-10-01 and validation thereafter. Empty stores at input start;
no coverage threshold. The 92 missing days in 2014 remain excluded.

64 uniform candidates, full metric catalogue on training:

| Batch | Warm simulation + scoring seconds | Discharge MiB |
| --- | ---: | ---: |
| 1 | 0.136 | 0.098 |
| 8 | 0.094 | 0.780 |
| 32 | 0.091 | 3.121 |
| 64 | 0.092 | 6.242 |

Separate 64-candidate simulation: 0.039 s; metrics: 0.054 s. Setup/input read:
0.038 s. First simulation with Numba cache: 0.091 s; fresh-cache compilation
and simulation: 3.114 s. These are single samples, not confidence intervals.

Two local NSGA-II runs, seeds 101/102, population 16, three generations,
KGE_NP/NSE objectives, batch 32, up to three retained models each: **0.206 s**
warm end-to-end including optimizer and retained train/validation diagnostics.
Each run evaluated 64 optimization candidates. Retained 3 and 1 models, no
padding. Second run retained training scores 0.822/0.872. This demonstrates the
interface, not convergence or scientifically meaningful ensemble uncertainty.

Process peak RSS: 167 MiB cached, 299 MiB cold compilation. macOS
`resource.getrusage().ru_maxrss` includes interpreter, libraries, compilation and
prior batches, not isolated batch or machine memory. Discharge sizes exclude
sort/rank workspace, forcing, state and optimizer matrices. Cold-cache repeat
ran alongside tests, with warm calibration 0.216 s. No cloud or full campaign.

Tests independently check metric examples, Pareto/crowding, reproducibility,
bounds, invalid candidates, warm-up, missing/coverage masks, validation exclusion,
batch equivalence and per-run retention. Validation and exact commit are in PR.
