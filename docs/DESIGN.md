# hydrologeez — Design Spec

> Output of the design/grill session. This is the durable architectural reference
> for the project. The orchestrator and all planning agents treat this as the
> source of truth for *what* and *why*; the *how* is planned per-milestone.

## Identity

**Conceptual hydrological models expressed as differentiable state-space models in JAX.**

- PyPI / import name: `hydrologeez` (`import hydrologeez`).
- The existing Rust project `pydrology` (`/Users/nicolaslazaro/Desktop/work/pydrology`)
  is **retired** as a runtime, but kept as the **numerical oracle** for validation.

## Motivation (settled)

1. **Contributor accessibility** — target contributors are experienced Python
   developers (hydrology researchers), not Rust developers. Rust + PyO3 + trait
   generics + a proc macro is a wall for them.
2. **Autodiff** — keep the door open to hybrid DL + traditional hydrology
   (differentiable models, gradient calibration, sensitivity analysis). Not the
   primary goal, but a first-class architectural constraint.

Adoption and contributor reach are valued **over raw single-run latency**. Rust
is retired despite being faster per single run, because JAX recovers performance
via `jit`+`lax.scan` (near-native time loop) and *wins* on batched calibration
via `vmap` (whole population / many catchments in one compiled, GPU-able call).
The one accepted permanent loss is single-step debuggability (opaque XLA traces
vs. a Rust stack trace).

## Core abstraction — the State-Space Model (SSM)

Discrete-time state-space form, taken literally:

```
transition:   (state, forcing) -> (state, fluxes)   # fused; computes ALL internal fluxes
observation:  (state, fluxes)  -> observable        # pluggable; default = streamflow
run:          lax.scan(transition) over forcing     # the fold
```

- `state` (x) = stores (production, routing, exponential, UH/MAXBAS buffers, ...)
- `forcing` (u) = precip, PET, temperature
- `params` (θ) = x1..x6 (GR6J), HBV's 14 params, ...
- `fluxes` (y, internal) = the full internal flux set per step
- `observable` = what we compare to data (default streamflow; pluggable for
  snow cover / ET later)

This is identical in shape to `lax.scan`'s `(carry, x) -> (carry, y)`, so the
SSM frame and the JAX execution model are the same thing — no impedance mismatch.

Everything the project wants is a named operation on the SSM:

- calibration = parameter estimation of θ
- autodiff = ∂(observable)/∂θ (sensitivities, gradient calibration)
- hybrid DL = replace part of `transition` or `observation` with an NN
  (`eqx.Module` field) — fits naturally
- data assimilation (EnKF / particle filter) = enabled by the distinct
  observation operator

## Representation: Equinox

- **All PyTrees are Equinox modules.** A model **is** an `eqx.Module` whose
  fields are its params → `jax.grad(loss)(model)` differentiates through params.
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
  Pareto). Population is evaluated via `vmap` in one batched, jit-compiled call.
  → requires adding a **batched-evaluate hook** to `ctrl-freak` upstream
  (owned: hydrosolutions/ctrl-freak). Dependency comes from **PyPI** now.
- **Gradient-based (local + autodiff showcase):** `optax` / `jax.grad` through
  `transition` → `run` → loss, for fast single-objective calibration.
- **Metrics** (NSE, KGE, logNSE, PBIAS, RMSE, MAE) are **rewritten JAX-native**
  so they are differentiable. No Rust metrics.
- params PyTree ↔ flat array boundary via `ravel_pytree` / `eqx.partition`.

## Precision & validation

- **float64 is required** (long store accumulation + metric stability). Enforced
  by a **loud import-time check** of `jax.config` — **no silent global x64 flip**
  (respects other libraries in the user's process). Fail fast with the one-line
  fix if x64 is off.
- The Rust `pydrology` is a **reference oracle, not gospel**. Parity target:
  **within ~1e-4 relative**, not bit-exact. Where Rust is wrong, the JAX rewrite
  corrects it and updates the fixture **with a note**.
- Validation harness: generate `(forcing → fluxes/streamflow)` golden fixtures
  from the Rust implementation; every JAX model must reproduce them within
  tolerance before acceptance.

## Scope & sequence (current milestone)

**In scope: GR6J and HBV-Light only.** Both standalone, daily, both exercise the
masked-kernel pattern (GR6J UH, HBV MAXBAS). No coupling required.

1. **GR6J** — proves the pipeline **and** the masked-kernel UH design. Build in
   thin layers: get `transition` + `scan` + oracle parity green **before**
   wiring calibration, autodiff, CI, and publishing on top.
2. **HBV-Light** — proves the contract generalizes (2nd kernel via `MAXBAS`) and
   stresses calibration (14-D).

**Out of scope (for now):** GR2M, CemaNeige, GR6J–CemaNeige, glacier coupling.
Keep the base contract **open to SSM composition** (output-flux-as-input-forcing)
so coupling is not foreclosed later.

## Packaging & conventions

Repo already scaffolded; follow its conventions:

- Build backend: `uv_build`. src layout. `requires-python >=3.13`.
- **uv only** (`uv add` / `uv sync` / `uv run`). No pip/poetry/conda.
- `ruff` (format + lint) and `ty` (type check). Modern typing (`list[str]`,
  `str | None`; no `typing.List/Optional`).
- `bump-my-version bump patch` on every commit; tag `v$(...)`.
- Tests: `pytest` (+ doctest). Use library assertions
  (`numpy.testing.assert_allclose` for arrays; works on JAX arrays via
  `np.asarray`).
- Docs: mkdocs-material. `docs/contracts.md` defines the contributor contract.
- CI: GitHub Actions + PyPI Trusted Publishing (OIDC).
- Core deps: `jax`, `equinox`, `optax`, `ctrl-freak` (PyPI). `jax[cuda]` as an
  optional extra.

## Quality bar

JOSS-grade as the floor (oracle parity + tests + docs + CI + working example +
contribution guide). Paper venue deliberately deferred — quality bar holds
regardless.

## Open items (resolve at build-time, non-blocking)

- Exact params↔flat-array adapter for the ctrl-freak boundary.
- Gradient-promise level: write diff-clean (`jnp.where`, soft clamps), add
  gradient-finiteness smoke tests, advertise gradient calibration only where
  validated against the oracle.
