# Input paths with nonlinear upper groundwater

This reruns the bounded [input-path study](README.md) with the nonlinear upper
response. The original report, `initial-measurements.json`, and `measurements.json`
remain historical evidence for the former UZL/K0 formulation. Their claim of an
unchanged model applies to that study, not this update.

The new fixed response is PERC=1, Alpha=0.5, K1=0.1, K2=0.05, MAXBAS=2.5.
UZL=12 and K0=0.2 are removed; Alpha=0.5 is an engineering midpoint setting,
not calibrated or recommended. K1's numerical value remains 0.1, but its units
are now mm^-0.5 day^-1, not day^-1. Other fixed parameters remain TT=0,
CFMAX=2, CFR=0.05, CWH=0.1, SCF=1, FC=100, BETA=1, LP=0.5. All are within
the supplied search ranges (SCF is supplied SFCF). Initial states remain
snow/liquid=0, soil=50 mm, groundwater/routing=0. History remains disabled.

The same canonical raw archive is read in place, without copying or extraction.
The same 128 basin sample, 1990–1999 window, five selection cases, four parameter
copies, four repeats, one-thread limits and six storage candidates are retained.
Every input path checks exact prepared inputs, outlet and complete final memory
against its resident reference under the **new** formulation.

This is an input-path timing comparison, not a controlled numerical-accuracy
experiment or an HBV-light executable validation. Old versus new runtime cannot
isolate an equation effect: model equations and parameters differ and run/cache
conditions are not controlled across studies. Cache and memory limitations in
the original report still apply. Soil behavior is unchanged; numerical accuracy
requires separate investigation.

## Reproduce

```sh
uv run --group experiments --group acquisition python -m scripts.hbv_input_path --raw /absolute/canonical/data/camels-us/raw --output .hbv-input-data --evidence experiments/hbv-input-path/nonlinear-measurements.json
```

The current script produces the nonlinear formulation. To reproduce historical
measurements, use the script and model at PR #124's merge (`ac7b5bc`), not the
current script. Generated stores are disposable, not evidence.

## Results

All 280 records completed, including exact input/output/final-memory checks.
For the largest case (128 basins × 3652 days × four parameter copies), warm
median milliseconds across repeats 1–3, packed basin-major:

| Path | Load | Simulate (includes validation) | Total |
| --- | ---: | ---: | ---: |
| resident | — | 208.3 | — |
| numpy-time | 10.5 | 204.7 | 215.2 |
| numpy-basin | 6.4 | 208.1 | 215.1 |
| zarr-time | 55.5 | 202.1 | 257.6 |
| zarr-basin | 47.6 | 202.8 | 248.6 |
| parquet-files | 125.8 | 199.9 | 325.9 |
| parquet-consolidated | 145.2 | 196.2 | 341.1 |

Each column is its own median; medians need not add. Dense NumPy inputs remain
the lowest-load-cost measured choice for this repeated-local case. The small
simulation differences across readers are not evidence of a kernel speedup.
The full [measurement file](nonlinear-measurements.json) retains all five cases,
both runtime orders, first observed reads and repeats.

Preparation took 2.12s. The signature warm-up timer
(`compilation_seconds`) took 0.09s with an existing
Numba cache: it is not a fresh-compilation benchmark. Process peak RSS was
764.3MiB on macOS, including construction and JIT;
it is not per-reader allocation. Versions: Python 3.13.14, NumPy 2.5.4,
Numba 0.68.1, Arrow 26.0.1, Zarr 3.4.2. Full validation: 57 tests pass,
Ruff lint/format and ty pass.
