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

## 6. Oracle-parity acceptance bar

The retired Rust `pydrology` is the **numerical oracle, not gospel**. Every model
must reproduce committed golden fixtures within **~1e-4 relative**
(`numpy.testing.assert_allclose(rtol=1e-4, atol=1e-6)`), not bit-exact. Exactly-zero
or tiny series need the small `atol` so they do not fail on relative tolerance.

Coverage that must ship with non-vacuous Rust-oracle evidence:

- The canonical run fixture (all 20 fluxes + streamflow).
- The **exchange-active** fixture (`x2=1.0, x5=0.5`): exercises the groundwater
  exchange limb on both the negative-R routing-clamp steps and the non-clamp steps.
- The **multi-`x4` UH ordinate table** (built from the true Rust binding) across
  integer and non-integer `x4`.
- The **large-positive exponential-store softplus branch** (`AR = Exp/x6 > 7`,
  `QRExp = Exp + x6/exp(AR)`) and the **+33 AR clamp**, via the crafted single-step
  oracle artifact.

Where Rust is wrong, the JAX rewrite corrects it and regenerates the fixture **with
a note**; for a corrected model, its equations are the reference and its golden
`.npz` are regression snapshots of them. GR6J version-fidelity corrections applied:
(1) the UH routing convolution reads the delay-line head **after** the shift (same-
day ordinate-1 term; removes a spurious +1-day lag), matching airGR `MOD_GR6J`;
(2) `actual_exchange_total = actual_exchange_routing + actual_exchange_direct + F`
(= airGR `MISC(15) = AEXCH1 + AEXCH2 + EXCH`). The three GR6J run/step `.npz` are
now corrected-Python regression snapshots. HBV-Light version-fidelity corrections
applied: (1) the MAXBAS routing convolution reads the delay-line head **after** the
shift (same-day ordinate-1 term; removes a spurious +1-step lag), sharing
`hydrologeez.convolution.convolve_delay_line` with GR6J (Seibert & Vis 2012);
(2) soil moisture above field capacity is routed to upper-zone recharge instead of
being discarded (mass-conserving; Seibert & Vis 2012), so the reported `recharge`
flux is the total soil->upper-zone flux (base recharge + FC overflow). The HBV run
`.npz` (canonical, maxbas25, overflow) are now corrected-Python regression
snapshots; `hbv_triangular_weights.npz` is UNCHANGED (no equation touches the
MAXBAS weights). With this, **both models' corrected Python equations are the
oracle** — no model remains a pending-audit Rust-parity snapshot.

Kept-by-decision (documented, not changed): GR6J parameter bounds use the internal
CODE convention (e.g. x6 ∈ [1, 50]), not the alternative published bounds; the HBV
explicit-split store over-draw follows standard forward-Euler and is retained.

## 7. Tooling

- `uv` only (`uv add` / `uv sync` / `uv run`). No pip/poetry/conda.
- `uv run ruff format` + `uv run ruff check` (lint), `uv run ty check` (types).
- Modern typing: `list[str]`, `str | None`; no `typing.List`/`Optional`.
- Tests use library assertions (`numpy.testing.assert_allclose`, works on JAX
  arrays via `np.asarray`).
- Every commit bumps the patch version (`uv run bump-my-version bump patch`) and is
  tagged `v$(uv run bump-my-version show current_version)`.
