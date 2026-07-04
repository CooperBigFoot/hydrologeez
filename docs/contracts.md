# Contributor contract

This page is the binding contract for adding or modifying a model in hydrologeez.

## 1. Implement exactly two methods

A model is a subclass of `hydrologeez.ssm.StateSpaceModel` (an `eqx.Module`). Its
fields are its calibratable parameters. You implement only:

- `init_state(self) -> State` - the initial `lax.scan` carry (the model stores).
- `transition(self, state, forcing) -> (state, fluxes)` - one fused step that
  returns the next state and all internal fluxes for that step.

You do not implement the time loop. The base class provides:

- `run(forcing, *, observation_operator=default_streamflow_observation, return_fluxes=False)`
  - folds `transition` over `forcing` with `lax.scan`. Returns the observable
  timeseries; with `return_fluxes=True` returns `(observable, fluxes, final_state)`.
- `batch_run(forcings, ...)` - `vmap` of `run` over a leading batch axis.

## 2. Process math lives in free functions

All process equations (production store, percolation, unit-hydrograph convolution,
routing, exchange, ...) live as plain free functions in a per-model `processes.py`,
so they are unit-testable in isolation and reusable. `transition` only wires them
together. Do not bury process math inside the module methods.

## 3. The observation operator is pluggable

An observation operator maps `(state, fluxes) -> observable`. The single canonical
default is `hydrologeez.default_streamflow_observation` (returns `fluxes.streamflow`).
Reference it; do not define a second, contradictory default. `run`/`batch_run`
accept an `observation_operator=` override.

## 4. Static-shape / masked-kernel policy (hard JAX rule)

Under `jit`/`vmap`, array shapes must be static. Therefore:

- **Structural integers** that set an array size (elevation bands, HBV zones, the
  GR6J UH length `nh`) are `eqx.field(static=True)` config fields, set at
  construction and **never calibrated**.
- **Continuous shape-affecting parameters** (GR6J `x4` -> UH length) use
  **fixed-length masked kernels** sized to a declared upper bound, with ordinates
  computed as a *smooth* function of the parameter and zero-padded/masked beyond
  active support. This keeps shapes static and keeps the parameter differentiable.
  GR6J ships `UH1` length 20 and `UH2` length 40 with `x4_max = 10`; the S-curves
  taper smoothly so `jax.grad` of the ordinates w.r.t. `x4` is finite and
  continuous across integer `x4`.

## 5. float64 requirement and enablement contract

float64 is required process-wide (long store accumulation + metric stability).

- hydrologeez enforces this with a **loud import-time check**: `import hydrologeez`
  calls `enforce_float64()`, which **raises** if `jax.config.jax_enable_x64` is not
  `True`. It deliberately does not silently flip `jax.config`.
- **Enablement is the caller's responsibility, and must happen before jax is
  imported**. JAX reads `JAX_ENABLE_X64` at import time. The three canonical
  mechanisms are:
  - Tests: `tests/conftest.py` sets `os.environ["JAX_ENABLE_X64"] = "1"` before any
    jax/hydrologeez import. Reuse it; do not invent another.
  - CI: the workflow declares `env: JAX_ENABLE_X64: "1"`.
  - Scripts/examples: set `os.environ["JAX_ENABLE_X64"] = "1"` at the very top,
    before importing jax/hydrologeez (`# noqa: E402` on the post-env imports).

One-line fix if you see the raise: run with `JAX_ENABLE_X64=1` set in the
environment before import.

## 6. Tooling

- `uv` only (`uv add` / `uv sync` / `uv run`). No pip/poetry/conda.
- `uv run ruff format` + `uv run ruff check` (lint), `uv run ty check` (types).
- Modern typing: `list[str]`, `str | None`; no `typing.List`/`Optional`.
- Tests use library assertions (`numpy.testing.assert_allclose`, works on JAX
  arrays via `np.asarray`).
- Every commit bumps the patch version (`uv run bump-my-version bump patch`) and is
  tagged `v$(uv run bump-my-version show current_version)`.

## 7. HDX I/O contract

The model kernel remains format-agnostic. Core hydrologeez imports and numerical
model execution must not import `polars`; HDX dependencies are loaded only through
the optional I/O entry points.

HDX is role-opaque, so hydrologeez owns the vocabulary and roles for columns it
understands. The default canonical dynamic fields are `precip`, `pet`, and `temp`
as forcing fields and `streamflow` as the target. Foreign column names must be
adapted through `Vocabulary` overrides instead of changing model forcing classes.

`from_hdx` and `to_hdx` must round-trip hydrologeez streamflow predictions through
HDX 0.2 scalar datasets. The writer emits string `basin_id` values, sorted
`datetime64[us]` times, per-basin `scalar_dynamic.parquet`, root
`scalar_static.parquet`, and a six-field manifest containing `format_version`,
`name`, `created_at`, `producer_version`, `crs`, and `cadence`.
