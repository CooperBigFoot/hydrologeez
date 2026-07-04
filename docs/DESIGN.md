# hydrologeez — Design Spec

> Living architectural reference for the project. The orchestrator and all
> planning agents treat this as the source of truth for *what* and *why*; the
> *how* is planned per-milestone. This document is no longer a porting brief —
> the initial port is complete and that framing is retired.

## Identity

**Differentiable conceptual hydrological models expressed as state-space models
in JAX.**

- PyPI / import name: `hydrologeez` (`import hydrologeez`).
- Substrate: `jax` + `equinox`. A model *is* an `eqx.Module` whose fields are its
  parameters.
- Correctness standard: the **documented per-model equations are the
  specification** (see [gr6j.md](gr6j.md), [hbv.md](hbv.md)).

## Motivation

Stated positively (not as a migration away from anything):

1. **A clean, tested, differentiable SSM library** for conceptual rainfall-runoff
   models, accessible to Python hydrology researchers (no Rust / PyO3 / trait
   generics wall for contributors).
2. **Batched calibration is the headline capability.** Whole populations / many
   catchments are evaluated in **one `vmap`-compiled call**, `jit`-compiled and
   GPU-able — the thing the substrate makes cheap that a per-run implementation
   does not.
3. **Differentiability is a first-class architectural property**, kept as an open
   door: `jax.grad(loss)(model)` differentiates straight through the time loop.
   Hybrid DL and gradient calibration are **architecturally supported but not yet
   exercised** — no hybrid model has been built, and the docs do not advertise
   one.

The one accepted permanent cost of the substrate is single-step debuggability
(opaque XLA traces vs. an eager stack trace).

## Core abstraction — the State-Space Model (SSM)

A conceptual rainfall-runoff model is a **nonlinear discrete-time state-space
model** — a dynamical system observed through a measurement map.

```
state (stores):     x_t ∈ R^n
forcing (inputs):   u_t ∈ R^m      (precip, PET, temperature)
parameters:         θ   ∈ R^p

transition:   (x_{t+1}, z_t) = f_θ(x_t, u_t)     z_t ∈ R^k internal fluxes
observation:  y_t = h(x_t, z_t)                  default h ⇒ streamflow
trajectory:   {y_t} = scan(f_θ) over u_{0:T-1}, from x_0
```

So `run` is a deterministic map `(θ, x_0, u_{0:T-1}) ↦ {y_t}`. This is identical
in shape to `lax.scan`'s `(carry, x) -> (carry, y)`, so the SSM frame and the JAX
execution model are the same thing — no impedance mismatch.

Every project verb is then a **named operation on that map**:

- **calibration** = `θ* = argmin_θ L({y_t(θ)}, y_obs)` — system identification.
- **sensitivity / gradient calibration** = `∂y_t/∂θ` via reverse-mode autodiff
  through the scan. *(Supported; not yet exercised.)*
- **hybrid DL** = replace part of `f_θ` or `h` with a network (`eqx.Module`
  field). *(Supported; not yet exercised.)*
- **data assimilation** = add process/observation noise
  `x_{t+1}=f_θ(x_t,u_t)+w_t`, `y_t=h(x_t,z_t)+v_t` and filter — enabled by the
  explicit observation operator `h`. *(Future.)*

Per-step symbols: `x` = stores (production/routing/exponential, UH / MAXBAS
buffers, ...); `u` = precip, PET, temperature; `θ` = x1..x6 (GR6J), the 14 HBV
params; `z` = the full internal flux set per step; `y` = the observable (default
streamflow; pluggable for snow cover / ET later).

## Representation: Equinox

- **All PyTrees are Equinox modules.** A model **is** an `eqx.Module` whose fields
  are its params → `jax.grad(loss)(model)` differentiates through params.
- An abstract base `eqx.Module` provides `run` (`lax.scan`) and a batched `run`
  (`vmap`) for free.
- A contributor implements only:
  - `init_state(self) -> State`
  - `transition(self, state, forcing) -> (state, fluxes)`
- Process math (production store, percolation, UH/MAXBAS convolution, ...) lives
  in plain **free functions** in a `processes.py` per model — unit-testable in
  isolation (supports regression-proof-before-fix discipline).

## Static-shape policy (hard JAX rule)

Under `jit`/`vmap`, array shapes must be static. Several params control sizes, so:

- **Structural integers** (e.g. elevation bands, HBV zones) → `eqx.field(static=True)`
  config fields, set at construction, **never calibrated**.
- **Continuous shape-affecting params** (GR6J `x4` → UH length; HBV `MAXBAS` →
  routing kernel length) → **fixed-length masked kernels** sized to a declared
  per-param upper bound, with ordinates computed as a smooth function of the
  param and zero-padded/masked beyond active support. Result: shapes are static
  **and** these params become differentiable (recovers the gradient otherwise
  lost to a discrete count).

## Calibration — dual stack

- **Derivative-free (global + multi-objective):** `ctrl-freak` (GA + NSGA-II
  Pareto). Population is evaluated via `vmap` in one batched, jit-compiled call
  through `ctrl-freak`'s `evaluate_batch` hook. Dependency comes from **PyPI**.
  **Shipped:** public `calibrate_evolutionary` (GA) and `calibrate_nsga2` (Pareto)
  wire `ctrl-freak`'s bounded `sbx_crossover` / `polynomial_mutation` operators
  over the batched evaluator.
- **Gradient-based (local + autodiff showcase):** `optax` / `jax.grad` through
  `transition` → `run` → loss, for fast single-objective calibration. **Shipped**
  as `calibrate_gradient`.
- **Metrics** (NSE, KGE, logNSE, PBIAS, RMSE, MAE) are **JAX-native** so they are
  differentiable. **Shipped** in `hydrologeez.metrics`.
- params PyTree ↔ flat array boundary via `ravel_pytree` / `eqx.partition`.
  **Shipped** as the `ParamSpec` adapter (`params_to_array` / `array_to_model`,
  plus generic `model_to_flat` / `flat_to_model`).

## Precision (float64 required, numerically justified)

**float64 is required** — justified by numerics, not by matching any reference
implementation.

- A conceptual model folds storage over 10³–10⁴ daily steps. float32 (~7
  significant digits) accumulates drift in long store balances and loses
  precision in calibration-metric sums (catastrophic cancellation in NSE/KGE
  differences), yielding **subtly wrong calibrated parameters** that are
  near-impossible to debug post hoc.
- Enforced by a **loud import-time check** of `jax.config` that **raises** if
  `jax_enable_x64` is off. **No silent global x64 flip** (respects other
  libraries in the user's process). Fail fast with the one-line fix.

## Validation — the equations are the specification

- The **documented per-model equations are the specification.** Correctness means
  fidelity to them.
- **Version-fidelity to the published GR6J (airGR) and HBV-Light formulations has been audited and reconciled.** Three confirmed divergences were corrected: (1) a shared read-before-shift routing lag in both the GR6J unit-hydrograph and HBV MAXBAS convolutions is now read-after-shift (same-day ordinate-1 term; matches airGR `MOD_GR6J` and Seibert & Vis 2012), via one shared `convolve_delay_line` helper; (2) GR6J `actual_exchange_total` now includes the exponential-store exchange leg F (airGR `MISC(15) = AEXCH1 + AEXCH2 + EXCH`); (3) HBV above-field-capacity soil moisture is now routed to upper-zone recharge instead of discarded (mass-conserving; Seibert & Vis 2012). Two behaviors are **retained by decision and documented**: the explicit-split (forward-Euler) store over-draw, which standard HBV implementations share (correcting it would diverge from published HBV), and the internal parameter-bound convention. The GR6J "magic constants / clamps" were verified **verbatim against airGR** (no change). The corrected Python equations are the specification for both models. See `contracts.md`, `gr6j.md`, and `hbv.md`.

## I/O — HDX is the canonical input interface

hydrologeez ingests and emits **HDX** datasets (`../hdx` — a prescriptive,
cloud-optimized per-basin hydrology data interface). A lumped conceptual model is
inherently **all-scalar I/O**, so it touches only HDX's scalar quadrants:

| hydrologeez | HDX quadrant | encoding |
|---|---|---|
| forcing: `precip[T]`, `pet[T]`, `temp[T]` | `scalar · dynamic` `[T]` | `basin=<id>/scalar_dynamic.parquet` cols |
| observed streamflow `[T]` | `scalar · dynamic` `[T]` | same parquet |
| static params / drainage area | `scalar · static` `[]` | root `scalar_static.parquet` |
| **model output streamflow** | `scalar · dynamic` `[T]` | a prediction dataset is just an HDX dataset |
| `batch_run` / `vmap` batch axis | HDX **basin-first partitioning** | many `basin=<id>/` dirs |

- **Canonical, not coupled.** HDX is the canonical, documented, default way in —
  a loader (`from_hdx`) reads the scalar parquet and yields the array-native
  `Forcing` the numerical core consumes. `model.run(forcing)` still takes plain
  arrays; the `jit`/`vmap` kernel **never depends on a file format**. Only parquet
  is needed for the scalar path (polars/pyarrow); no zarr/COG.
- **Vocabulary on top of HDX.** HDX is role-agnostic by design (field names are
  opaque; it carries no forcing/target roles). hydrologeez owns the **semantic
  layer**: a documented vocabulary mapping field names → model roles (defaults
  `precip`/`pet`/`temp` → forcing, `streamflow` → target), plus an explicit
  field-role map override for foreign datasets. Factor into a shared HDX profile
  only when a second consumer needs it.
- **Predictions are HDX.** Model output is written back as a conformant
  `scalar·dynamic` prediction dataset.
- **Batching.** HDX basin-first partitioning ↔ the `vmap` batch axis; many basins
  → `batch_run`. Ragged per-basin time records are reconciled by **pad+mask** (or
  equal-length grouping), since `vmap` requires static shapes.
- `hdx-core` reads **metadata only** (`validate`/`describe` over footers/schemas);
  hydrologeez reads the column values itself and may call `validate` first to fail
  fast on a non-conformant dataset.

### Future: gridded → scalar derivation edge (out of scope now)

When a model needs a **DEM** (elevation bands, hypsometric curve) the DEM ships
via HDX as `gridded·static` (COG). It is read **once at preprocessing** and
reduced to **structural integers** (band count → `eqx.field(static=True)`) and
`scalar·static` per-band attributes; the differentiable model **never
differentiates through a raster**. This adds a raster-reader dependency and lands
only when multi-zone HBV / CemaNeige do. HDX is simply the transport for the DEM.

## Scope & sequence

**Done (step one — the port):** GR6J and HBV-Light, both standalone, daily, both
exercising the masked-kernel pattern (GR6J UH, HBV MAXBAS). Both have transition +
scan implementations.

**Out of scope (for now):** GR2M, CemaNeige, GR6J–CemaNeige, glacier coupling,
multi-zone (elevation-band) HBV. Keep the base contract **open to SSM composition**
(output-flux-as-input-forcing) so coupling is not foreclosed later.

**Next milestones (candidates):** the HDX I/O layer (canonical scalar ingestion +
prediction writer + vocabulary). The version-fidelity audit (GR6J vs airGR, HBV vs HBV-Light) is now **complete** — see the Validation section. Calibration
wiring — the `ctrl-freak` batched hook, the `optax` gradient path, JAX-native
metrics, the params↔array adapter, and the public
`calibrate_evolutionary`/`calibrate_nsga2` API — is now **shipped**.

## Packaging & conventions

Repo already scaffolded; follow its conventions:

- Build backend: `uv_build`. src layout. `requires-python >=3.11`.
- **uv only** (`uv add` / `uv sync` / `uv run`). No pip/poetry/conda.
- `ruff` (format + lint) and `ty` (type check). Modern typing (`list[str]`,
  `str | None`; no `typing.List/Optional`).
- `bump-my-version bump patch` on every commit; tag `v$(...)`.
- Tests: `pytest` (+ doctest). Use library assertions
  (`numpy.testing.assert_allclose` for arrays; works on JAX arrays via
  `np.asarray`).
- Docs: mkdocs-material. `docs/contracts.md` defines the contributor contract.
- CI: GitHub Actions + PyPI Trusted Publishing (OIDC).
- Core deps: `jax`, `equinox`, `optax`, `ctrl-freak` (PyPI). HDX scalar ingestion
  adds a parquet reader (polars/pyarrow). `jax[cuda]` as an optional extra.

## Quality bar

JOSS-grade as the floor (equation-fidelity + tests + docs + CI + working example +
contribution guide). Paper venue deliberately deferred — quality bar holds
regardless.

## Open items (resolve at build-time, non-blocking)

- Gradient-promise level: write diff-clean (`jnp.where`, soft clamps), add
  gradient-finiteness smoke tests, advertise gradient calibration only where
  validated.
- HDX vocabulary: hydrologeez-owned convention now; shared profile later if a
  second consumer needs it.
