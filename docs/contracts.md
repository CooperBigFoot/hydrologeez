# Contributor contract

This page is the binding contract for adding or modifying a hydrologeez model.

## 1. State-space model interface

A model subclasses `hydrologeez.ssm.StateSpaceModel`, an abstract
`torch.nn.Module`, and implements exactly two methods:

```python
def init_state(self, parameters, *, batch_size): ...

def transition(self, state, forcing, parameters): ...

model.run(
    forcing,
    parameters=None,
    warmup=None,
    warmup_parameters=None,
    observation_operator=default_streamflow_observation,
    return_fluxes=False,
)
```

`init_state` receives the resolved parameter mapping and batch size. `transition`
receives one timestep of batched forcing and the parameter tensors, and returns
the next state and fluxes. The base `run` method performs the eager Python time
loop. It uses registered parameters when `parameters` is omitted; otherwise it
uses the supplied complete mapping. It returns observations `[B, T]`, or
`(observations, fluxes, final_state)` when `return_fluxes=True`, with dataclass
flux leaves stacked on dimension 1.

## 2. Process and observation functions

Process equations belong in plain free functions in each model's `processes.py`,
where they are independently testable and reusable. `transition` wires them
together. The pluggable observation operator maps `(state, fluxes)` to an
observable; the canonical default is
`hydrologeez.default_streamflow_observation`.

## 3. Tensor shapes and fixed kernels

All forcing leaves share `[B, T]`. Per-step state and flux tensor leaves preserve
the leading batch dimension. Explicit parameter leaves may be scalar `[]`,
per-basin `[B]`, or per-basin/per-time `[B, T]`.

GR6J unit-hydrograph and HBV routing delay lines remain fixed-size. This preserves
model semantics and consistent batched tensor shapes while their ordinates remain
tensor functions of continuous parameters. Structural integers are ordinary,
non-calibrated module configuration.

## 4. Dtype and device

Float64 on CPU is the reference and golden-fixture path. Float32 on an explicitly
selected accelerator is the training path. Use `reference_tensor`,
`training_tensor`, `reference_defaults`, and `training_defaults` for explicit
local choices. Preserve caller dtype/device. Do not call
`torch.set_default_dtype`, mutate the default device, or add an import-time guard.

## 5. Warmup

Warmup is a separate forcing container with the same batch size and may be empty.
Its transitions run under `torch.no_grad()` and its final state is detached before
the main period creates its own graph. `warmup_parameters` may be supplied
separately; otherwise the resolved main parameters are reused.

## 6. Explicit parameters and calibration

Registered `nn.Parameter` leaves are ordinary model state. Functional calibration
must pass a complete explicit mapping without mutating the template model during
objective evaluation. Scalar, per-basin, and per-basin/per-time mappings support
spatial and time-varying parameterization.

`calibrate_gradient` from `hydrologeez.calibration` uses a bounded sigmoid
transform, explicit mappings, `torch.optim`, and optional SSM warmup forcing. It
returns a calibrated model and detached loss history. Evolutionary GA/NSGA-II
keeps ctrl-freak's NumPy population/results boundary but evaluates each population
in one batched Torch call under `torch.no_grad()`. Its integer `warmup` slices the
objective period and is distinct from SSM warmup forcing.

## 7. Tooling and releases

- Use `uv` only; do not use pip, poetry, conda, or pip-tools.
- Run `ruff` formatting/linting, `ty` type checking, and `pytest`.
- Use modern typing (`list[str]`, `str | None`).
- Use `numpy.testing` for NumPy leaves and `torch.testing.assert_close` for Torch
  tensors; retain xarray and polars testing utilities for those objects.
- Versions are bumped only in release commits through `bump-my-version`, never in
  ordinary commits. Publishing and tags are created only by GitHub Releases/OIDC.

## 8. HDX I/O

The model core is format-agnostic. Optional Parquet dependencies load lazily
through HDX entry points. HDX is role-opaque, so hydrologeez owns the canonical
`precip`, `pet`, `temp`, and `streamflow` vocabulary and foreign-name overrides.

`from_hdx` returns NumPy forcing-dictionary leaves, streamflow, statics, mask, and
NumPy times. `.torch(dtype=..., device=...)` is the explicit bridge that constructs
the selected model forcing dataclass and converts the mask to `torch.bool`.
Padding and masks represent ragged multi-basin data. `to_hdx` must preserve the
HDX 0.2 round-trip contract: string basin IDs, sorted `datetime64[us]` times,
per-basin dynamic files, root statics, and the six-field manifest.

## 9. hcx boundary

Adapters in `hydrologeez.hcx` exist only for development conformance. They import
hcx lazily during forecast creation, consume only `scalar_dynamic`, and preserve
point metadata. Do not import them from the default package path or expose package
entry points. hydrologeez must not register an `hcx.models` plugin.
