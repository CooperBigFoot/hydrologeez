# hydrologeez

hydrologeez provides differentiable GR6J and HBV state-space models as PyTorch
`nn.Module` objects. An eager time loop makes each transition directly
debuggable while Torch autograd differentiates through the simulation.

## State-space contract

```text
transition:  (state[B,...], forcing[B,...], parameters) -> (state, fluxes)
observation: (state, fluxes) -> observable[B]
run:         eager fold over forcing[:, time] -> observable[B, T]
```

- Forcing and returned observations have leading `[batch, time]` dimensions.
- Registered `nn.Parameter` values are the default; complete explicit mappings
  support scalar, per-basin, and per-basin/per-time values.
- Warmup uses separate forcing under `torch.no_grad()` and detaches its final
  state before the main simulation.
- Gradient calibration uses `torch.optim`; GA and NSGA-II evaluate a whole
  population as one no-grad Torch batch.
- HDX loading is NumPy-first, with an explicit dtype/device conversion to Torch.
- `hydrologeez.hcx` is a development-only, lazy conformance adapter and publishes
  no `hcx.models` entry point.

## Dtype and device

Use float64 on CPU for references and reproducibility, and float32 on an
explicitly selected accelerator for training.

```python
from hydrologeez import reference_tensor, training_tensor

cpu_reference = reference_tensor([1.0, 2.0])
accelerator_training = training_tensor([1.0, 2.0], device="cpu")
```

Production accelerator code passes its actual CUDA or MPS device instead of
`"cpu"`. These helpers make local conversions: importing hydrologeez performs no
precision check and does not mutate process-global Torch defaults.

## Where to go next

- [GR6J](gr6j.md): equations, state, parameters, and usage.
- [HBV](hbv.md): the single-zone HBV-Light implementation.
- [HDX](hdx.md): NumPy-first dataset loading and explicit Torch conversion.
- [Contributor contract](contracts.md): tensor shapes, calibration, tooling, and
  architectural boundaries.
- [Home](index.md): this overview.
