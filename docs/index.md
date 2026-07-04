# hydrologeez

**Differentiable conceptual hydrological models expressed as state-space models in JAX.**

hydrologeez reimplements conceptual rainfall-runoff models (GR6J and HBV-Light) as
[Equinox](https://docs.kidger.site/equinox/) modules over a discrete-time
state-space form, executed with `jax.lax.scan`.

A model *is* an `eqx.Module` whose fields are its parameters, so
`jax.grad(loss)(model)` differentiates straight through the time loop. Whole
populations / many catchments are evaluated in one `vmap`-compiled call for
calibration.

## The state-space form

```
transition:  (state, forcing) -> (state, fluxes)   # fused; computes ALL internal fluxes
observation: (state, fluxes)  -> observable        # pluggable; default = streamflow
run:         lax.scan(transition) over forcing     # the fold
```

## Precision is required and enforced

float64 is mandatory (long store accumulation + metric stability). hydrologeez
performs a **loud import-time check** and *raises* if JAX x64 is off. It never
silently flips global config. Enable it **before importing jax or hydrologeez**:

```bash
JAX_ENABLE_X64=1 python your_script.py
```

or in-process, before any jax import:

```python
import os
os.environ["JAX_ENABLE_X64"] = "1"
import jax  # noqa: E402
```

## Where to go next

- [GR6J model](gr6j.md) - the equations, usage, and the API reference.
- [HBV model](hbv.md) - the 14-parameter HBV-Light single-zone model: snow/soil/response math, the MAXBAS masked routing kernel, and the API reference.
- [Contributor contract](contracts.md) - the two-method model contract, the static-shape/masked-kernel policy, the float64 enablement contract, tooling conventions (formatting, typing, testing), and the HDX I/O contract.
