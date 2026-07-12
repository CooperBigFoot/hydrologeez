# hydrologeez

[![PyPI version](https://img.shields.io/pypi/v/hydrologeez)](https://pypi.org/project/hydrologeez/)
[![Python versions](https://img.shields.io/pypi/pyversions/hydrologeez)](https://pypi.org/project/hydrologeez/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Differentiable conceptual rainfall-runoff state-space models implemented as
PyTorch `torch.nn.Module` objects. GR6J and HBV run an eager recurrence over
time, with Torch autograd available through the complete simulation.

## Installation

```bash
# uv (recommended)
uv add hydrologeez

# pip
pip install hydrologeez
```

With optional HDX support:

```bash
uv add "hydrologeez[hdx]"
```

## Dtype and device policy

Use `torch.float64` on CPU for numerical references and reproducibility. Use
`torch.float32` on an explicitly selected accelerator for training. Conversion
helpers make those choices locally and never mutate Torch global defaults.

```python
import torch

from hydrologeez import reference_tensor, training_tensor

reference = reference_tensor([1.0, 2.0, 3.0])
training = training_tensor([1.0, 2.0, 3.0], device="cpu")

assert reference.dtype == torch.float64
assert training.dtype == torch.float32
```

## Quick start

Forcing tensors use leading `[batch, time]` dimensions. This one-basin example
therefore returns shape `[1, 10]`.

```python
import torch

from hydrologeez.models.gr6j import GR6J, GR6JForcing

dtype = torch.float64
model = GR6J(
    x1=torch.tensor(350.0, dtype=dtype),
    x2=torch.tensor(0.0, dtype=dtype),
    x3=torch.tensor(90.0, dtype=dtype),
    x4=torch.tensor(1.7, dtype=dtype),
    x5=torch.tensor(0.3, dtype=dtype),
    x6=torch.tensor(5.0, dtype=dtype),
)
forcing = GR6JForcing(
    precip=torch.tensor([[0.0, 5.0, 12.0, 8.0, 0.0, 0.0, 3.0, 20.0, 1.0, 0.0]], dtype=dtype),
    pet=torch.tensor([[1.0, 1.2, 1.1, 0.9, 1.0, 1.3, 1.1, 0.8, 1.0, 1.2]], dtype=dtype),
)

streamflow = model.run(forcing)
print(streamflow.shape)  # torch.Size([1, 10])
```

Pass `return_fluxes=True` to inspect the full trajectory:

```python
streamflow, fluxes, final_state = model.run(forcing, return_fluxes=True)
```

Tensor leaves in the flux dataclass are stacked as `[B, T]`.

## Features

- GR6J and single-zone HBV rainfall-runoff models.
- Torch autograd through the eager recurrence.
- Native leading batch dimensions on forcing, observations, state, and fluxes.
- Registered parameters or explicit scalar, per-basin, and per-time tensors.
- Separate no-grad warmup with a detached initial state for the main period.
- Gradient calibration with `torch.optim` and evolutionary GA/NSGA-II calibration.
- Explicit, local dtype and device selection.

Gradient calibration uses bounded functional parameter mappings with
`torch.optim`. Evolutionary calibration retains ctrl-freak's NumPy boundary but
evaluates each population in one batched Torch call under `torch.no_grad()`.
See the [contributor contract](https://cooperbigfoot.github.io/hydrologeez/contracts/)
and [model documentation](https://cooperbigfoot.github.io/hydrologeez/).

## The name

hydrologeez is **hydrology** + **"geez"**, the exclamation. That is all there is to it.

## Links

- [Documentation](https://cooperbigfoot.github.io/hydrologeez/)
- [Changelog](CHANGELOG.md)
- [License (MIT)](LICENSE)
