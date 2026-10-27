# HBV execution experiments

Experimental code only; no production model or public API is adopted. This returns
execution evidence to [#99](https://github.com/CooperBigFoot/hydrologeez/issues/99)
and the later design discussion [#91](https://github.com/CooperBigFoot/hydrologeez/issues/91).

## Reproduce

```sh
uv sync --group experiments
uv run --group experiments pytest
uv run ruff check experiments tests
uv run ty check
uv run --group experiments python -m experiments.hbv.benchmark --output experiments/hbv/evidence/serial.json
uv run --group experiments python -m experiments.hbv.benchmark --backend parallel --output experiments/hbv/evidence/parallel.json
uv run --group experiments python -m experiments.hbv.benchmark --backend expanded --output experiments/hbv/evidence/expanded.json
uv run --group experiments python -m experiments.hbv.benchmark --basins 1000 --candidates 4 --days 90 --history --output experiments/hbv/evidence/history.json
uv run --group experiments python -m experiments.hbv.benchmark --basins 1000 --candidates 4 --days 90 --disk .worktrees/hbv-output --output experiments/hbv/evidence/disk.json
```

`--disk` writes disposable generated data, not evidence. JSON timings are evidence.
Use separate processes and do not run workloads concurrently. Numba serial warm-up
loads an existing cache when present; startup is not necessarily cold compilation.
Parallel compilation is uncached: compiling the same function with two cached Numba
decorators initially reused the serial variant, an unsuccessful measurement corrected
before final results. No fastmath is used. Each calculation repeats three times;
throughput uses median elapsed time and counts simulation-days, not calibrations.

## Resolved input semantics

The only model is the exact ordered daily recipe in [the reference](../../docs/hbv-light.md),
not verified HBV-light executable equivalence. Array contracts are experiments, not
storage/API commitments. `inputs.synthetic` explicitly supplies every parameter and
initial state. All computations assume valid finite domain inputs; there is no parser,
validation framework, sampler API or calibration engine. Units are mm daily amounts,
°C daily mean temperature, and day-based rates. Day 0 begins an artificial daily
sequence; all three forcing channels cover the same intervals. No calendar or real
data are implied. No scientific defaults were inferred from illustrative TOML.

Seed 20261027; each basin cycles through 1,2,3,4 occupied zones with positive random
area fractions normalized to one. Four physical slots are retained, inactive slots
have zero area and are excluded from physical interpretation. Their parameters remain
valid so vectorized calculations do not divide by zero. Routing similarly pads to
four integrated daily bins with zeros, never rounds fractional MAXBAS.

Forcing layout is basin × day × zone × (P,T,PET), float64 on CPU. Candidates refer
to forcing/area rows by `forcing_index`; `basin_index` always retains physical basin
identity for candidate and fixed-value associations. The expanded alternative
materializes candidate copies and changes only the execution lookup index. Rain
is gamma(shape=.7, scale=6) with 45% occurrence. Temperature is an annual sinusoid
(amplitude 8°C), Gaussian daily noise (mean 1, sd 3°C), minus zone index. PET is
1.5 plus the same sinusoid, identical across zones by explicit assumption. These
synthetic values exercise snow, melt, wet/dry soil, recession and routing, not skill.

Parameters are reproducibly uniform within these **experimental**, not calibrated
or recommended, bounds: TT [-1,1], CFMAX [1,5], CFR [.02,.2], CWH [.02,.2], SCF
[.8,1.2], FC [40,250], BETA [1,3], LP [.3,1]; PERC [0,3], UZL [1,20], K0
[.05,.8], K1 [.02,.4], K2 [.005,.1], MAXBAS [1,4]. Parameters are fixed through
each trajectory. Zone initial S=5,L=0,M=.4FC; basin U=3,V=10, empty routing.
These are explicit experiment choices, not model constants. No warm-up is excluded:
the entire sequence is evaluated separately with each candidate's parameters.

The retained sharing map ties TT/CFR/SCF by basin; CFMAX/CWH/LP by vegetation
(zone index modulo 2); BETA by elevation (zone index divided by 2); FC by zone.
This extends parameterization in the HBV document, not process equations; tied
parameters never merge state. Basins cycle among no fixed entries, fixed TT/SCF/K2,
and all fixed. Fixed entries explicitly copy the first candidate's supplied values.
All-fixed basins intentionally repeat identical simulations, not optimization runs.
Changing a supplied fixed input changes the result. No parameter-dependent snow or
warm-up cache is reused. Fixed subsets alone do not guarantee a runtime speedup.

## Outputs, restart and bounded histories

`simulate` returns timestep outlet Q and final zone S,L,M and basin U,V,h memory.
With `history=True`, states include initial x0 and end-of-day x1…xT. Fluxes have
T rows: I,R,A,corrected P,melt,refreeze,raw recharge,overflow per zone;
Q,G,D,q0,q1,q2 per basin. Intermediate stores/demands and corrected snowfall/rain
can be reconstructed from these, the saved parameters, forcing and start states.
Basin R,A,P* are area-weighted zone transfers, not sums of zone depths. All state
and flux arrays retain simulation/zone/day indexes. Basin and candidate identities,
sharing maps, area and fixed flags remain in `Inputs`.

Parquet output is independent of equations and chunks **time**, retaining only a
bounded number of days' histories across simulations. `state_day=0` is initial;
rows n+1 contain state x[n+1] and the flux for interval [n,n+1). Initial fluxes
are null/NaN. `zone_id`, `basin_id`, `candidate`, and area fraction are explicit.
The setup identity is this single documented daily HBV experiment. `inputs.npz`
contains complete resolved inputs and forcing positions are `state_day`; a saved
state plus those inputs can resume with `simulate(start=state_day, initial=...)`.
Chunk boundaries never reset routing or warm-up. Restart and Parquet boundary
round-trip tests compare full state and routing, not only discharge. The initial
input archive write is outside timed Parquet costs; output timing separates model
history calculation, Arrow materialization, and compression/write/close. OS page
cache and lack of fsync mean these are application writing costs, not durable-media
bandwidth. No permanent storage contract or streaming API is selected.

## Local hardware and scope

Apple M4 Max MacBook Pro; 16 CPU cores (12 performance,4 efficiency), 40 GPU cores,
64 GiB unified memory, macOS 15.7.3 arm64, Metal 3. Python and versions are locked
in `uv.lock`. Numba uses 8 threads. CPU float64 and MLX GPU float32 are evaluated;
Metal does not support float64. No NVIDIA, rental, remote computation or real-data
skill assessment. CPU RSS is process high-water mark including preparation,
compilation, reference and previous result allocations, not isolated kernel memory.
Histories' RSS includes full reference materialization for fidelity. The separate
disk process does not allocate a full in-memory trajectory. GPU allocator and RSS
metrics are different and must not be added or directly interpreted as equal.

## Findings

| Workload / request | Method | Median seconds | Range seconds | Peak memory |
| --- | --- | ---: | ---: | ---: |
| 1000 basins × 8 candidates × 365 days / Q | serial CPU | 0.3943 | .3920–.3973 | 296 MB RSS |
| same / Q | 8-thread CPU | 0.05262 | .05184–.05292 | 415 MB RSS |
| same / Q | serial CPU expanded forcing | 0.3862 | .3829–.4080 | 556 MB RSS |
| same / Q | GPU | 0.1259 | see raw 5 repeats | 85 MB incremental Metal |
| 2000 × 16 × 365 / Q | 8-thread CPU | 0.2066 | .20655–.20663 | 762 MB RSS |
| 1000 × 4 × 90 / full history | serial CPU | 0.06089 | .05636–.06099 | 799 MB RSS |
| same / full history | 8-thread CPU | 0.00918 | see GPU comparison JSON | not isolated |
| same / full history | GPU | 0.03907 | see raw 5 repeats | 162 MB incremental Metal |

MB are decimal; peaks differ in scope as described above. The paired CPU/GPU
comparison is independently repeated in [GPU.md](GPU.md); its serial CPU numbers
vary slightly from standalone runs. All CPU comparisons have zero maximum absolute
error for every requested array against serial float64. Independent worked-day,
nonlinear soil, snow/refreeze, area/limited demands, fractional routing, full-memory
restart and conservation tests anchor correctness rather than relying on correlated
backend agreement. CPU tolerance in independent arithmetic cases uses NumPy
assert_allclose default relative 1e-7 and zero absolute unless explicit; cumulative
water accounting uses absolute 2e-13 mm. GPU comparison tolerance and actual errors
are in [GPU.md](GPU.md), including absolute errors and normalized tolerance ratios.

The larger measured workload sustains **56.54 million simulation-days/s** (32,000
candidate simulations across 2,000 basins), not projected throughput. Expanding forcing
multiplies its allocation from 35.04 to 280.32 MB, costs .014 s preparation and gives
no clear speed benefit over shared input. This is a useful negative result for
[#90](https://github.com/CooperBigFoot/hydrologeez/issues/90): preserve forcing reuse
and explicit simulation→basin indexing, while treating disk layout independently.
Padding four zones averages 2.5 active zones; packing could reduce inactive work but
was not measured. No claim is made for arbitrary zone counts or forcing readers.

CPU independent-simulation parallelism is the best measured approach here, not a
production selection. MLX GPU beats serial CPU but loses to 8-thread CPU on both
matched requests; per-day dispatch and gathering are plausible costs, not a proven
profile diagnosis. GPU float32 passed controlled and synthetic comparisons but has
no universal threshold-equivalence guarantee. A fused time-loop GPU kernel could be
worth a later bounded test with many more candidates; no performance is predicted.

For the same 1000×4×90 full-history request, bounded Parquet recording took .0591 s
calculation, .0228 s Arrow materialization, .4559 s compression/write/close; process
peak RSS was 575 MB versus 799 MB for in-memory correctness comparison. Chunk size
16 days means output buffers scale with chunk length rather than trajectory length;
forcing and current state still scale with workload. This is not an isolated memory
saving ratio because the in-memory run holds reference outputs too. Disk writing was
measured once, so its variability is unknown; no disk speedup is claimed. Exact
components are retained in [evidence/disk.json](evidence/disk.json). Model calculations
never call Parquet. Histories include fluxes as well as every routing entry; prediction
runs do not silently defer requested history work.

Precomputation in this experiment is limited to integrated routing weights, parameter
sharing resolution and reusable basin forcings, all included in preparation. No
snow/soil precomputation speedup is claimed; many candidates vary those dependencies.
Repeated evaluation assumes unchanged resident forcing and parameters and reruns
all days, including candidate-dependent initial evolution. Initial-state sensitivity,
real calibrations and warm-up convergence remain future scientific questions.

The scalar temporal feasibility probe and its limitations are reported separately
in [TEMPORAL.md](TEMPORAL.md). It is not an alternative whole-HBV performance result.

Evidence is bounded synthetic local measurement. It does not choose an implementation,
freeze a configuration/output contract, establish hydrological skill, or prove
HBV-light software equivalence. The next #91 discussion should weigh CPU simplicity,
GPU batch scaling, output requirements, precision and any need for more investigation.
