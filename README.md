# hydrologeez

Differentiable conceptual hydrological models in JAX.

[![PyPI version](https://img.shields.io/pypi/v/hydrologeez)](https://pypi.org/project/hydrologeez/)
[![Python versions](https://img.shields.io/pypi/pyversions/hydrologeez)](https://pypi.org/project/hydrologeez/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

hydrologeez implements conceptual rainfall-runoff models (GR6J, HBV) as
[Equinox](https://github.com/patrick-kidger/equinox) modules on top of
[JAX](https://github.com/google/jax). Every model is a PyTree whose fields are its
parameters, so a full simulation is end-to-end differentiable: `jax.grad` flows through
the entire `lax.scan` over your forcing series. That makes both gradient-based and
evolutionary parameter calibration first-class.

## Installation

```bash
# uv (recommended)
uv add hydrologeez

# pip
pip install hydrologeez
```

GPU (CUDA 12) wheels of JAX via the `cuda` extra:

```bash
uv add "hydrologeez[cuda]"
# or
pip install "hydrologeez[cuda]"
```

## Requirement: 64-bit precision (`JAX_ENABLE_X64=1`)

> **hydrologeez requires JAX 64-bit (float64) precision.** It does **not** flip this for
> you. You must enable x64 **before** importing `jax` or `hydrologeez`, otherwise the
> import raises `RuntimeError`.

Enable it either by exporting the environment variable before running Python:

```bash
export JAX_ENABLE_X64=1
```

or, in code, as the very first thing your program does:

```python
import os
os.environ["JAX_ENABLE_X64"] = "1"  # must run before any jax/hydrologeez import
```

## Quick Start

A minimal forward run: build a GR6J model from explicit parameters, feed it a short
synthetic daily forcing series, and read out simulated streamflow. No data files needed.

```python
import os

os.environ["JAX_ENABLE_X64"] = "1"  # must be set before importing jax/hydrologeez

import jax.numpy as jnp

from hydrologeez.models.gr6j import GR6J, GR6JForcing

# GR6J's six parameters (x1..x6), each a scalar JAX array.
model = GR6J(
    x1=jnp.asarray(350.0),  # production store capacity [mm]
    x2=jnp.asarray(0.0),    # groundwater exchange coefficient
    x3=jnp.asarray(90.0),   # routing store capacity [mm]
    x4=jnp.asarray(1.7),    # unit-hydrograph time base [days]
    x5=jnp.asarray(0.3),    # inter-catchment exchange threshold
    x6=jnp.asarray(5.0),    # exponential store depth [mm]
)

# Daily forcing; the leading axis is time. Precipitation and PET in mm/day.
precip = jnp.asarray([0.0, 5.0, 12.0, 8.0, 0.0, 0.0, 3.0, 20.0, 1.0, 0.0])
pet = jnp.asarray([1.0, 1.2, 1.1, 0.9, 1.0, 1.3, 1.1, 0.8, 1.0, 1.2])
forcing = GR6JForcing(precip=precip, pet=pet)

# Forward run -> simulated streamflow timeseries, shape (T,), dtype float64.
streamflow = model.run(forcing)
print(streamflow)
```

For full internal fluxes, pass `return_fluxes=True`:
`observable, fluxes, final_state = model.run(forcing, return_fluxes=True)`.

Gradient-based and evolutionary calibration (using ctrl-freak's `ga` / `nsga2` with the
`evaluate_batch` hook) are covered in the [documentation](https://cooperbigfoot.github.io/hydrologeez/).

## Features

- **GR6J and HBV** conceptual rainfall-runoff models, ready to run.
- **Fully differentiable** in JAX: each model is an Equinox PyTree, so `jax.grad` and
  `jax.value_and_grad` flow through the whole `lax.scan` simulation.
- **Two calibration paths**: gradient descent (e.g. `optax`) and evolutionary search
  (`ga` / `nsga2` from ctrl-freak, with a batched `evaluate_batch` hook).
- **Validated numerics**: streamflow matches the retired Rust `pydrology` oracle to
  ~1e-4 relative error.
- **Batched simulation** via `batch_run` (vmap over a leading batch axis).
- **64-bit by default**: import-time x64 enforcement keeps long store recurrences and
  metric reductions numerically stable.

## Links

- [Documentation](https://cooperbigfoot.github.io/hydrologeez/)
- [Changelog](CHANGELOG.md)
- [License (MIT)](LICENSE)
