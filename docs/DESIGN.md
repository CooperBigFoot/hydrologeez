# hydrologeez — Design Spec

> Living architectural reference for the project's purpose, contracts, and
> scientific decisions. Per-model equations remain the correctness specification.

## Identity and motivation

hydrologeez implements differentiable conceptual hydrological state-space models
as PyTorch `nn.Module` objects. GR6J and single-zone HBV provide eager,
single-step debuggability, native batch tensors, Torch autograd, gradient
calibration, and evolutionary GA/NSGA-II calibration.

The library targets hydrology researchers with an ordinary Python implementation
and explicit state, flux, and observation contracts. A complete evolutionary
population is evaluated in one batched model call. Hybrid models remain a possible
future composition, not a shipped feature.

## Core abstraction — state-space model

```text
state (stores):     x_t ∈ R^n
forcing (inputs):   u_t ∈ R^m
parameters:         θ   ∈ R^p

transition:   (x_{t+1}, z_t) = f_θ(x_t, u_t)
observation:  y_t = h(x_t, z_t)
trajectory:   {y_t} = eager fold of f_θ over u_{0:T-1}, from x_0
```

Forcing tensor leaves are `[B, T, ...]`; each transition receives `[B, ...]`.
`run` returns observations `[B, T]`, and stacks dataclass flux leaves on time
dimension 1. Calibration identifies θ by minimizing an objective over this map;
data assimilation remains future work enabled by the explicit observation map.

## Representation: PyTorch

`StateSpaceModel` is an abstract `torch.nn.Module`. Subclasses implement:

- `init_state(parameters, *, batch_size)`
- `transition(state, forcing, parameters)`

The base `run` performs an eager Python loop. Registered `nn.Parameter` leaves are
the fallback when no mapping is supplied. Complete explicit mappings accept
scalar `[]`, per-basin `[B]`, or per-basin/per-time `[B,T]` tensors, enabling
functional calibration and spatial or time-varying parameterization without
mutating the module. Process equations remain free functions in each model's
`processes.py`.

## Fixed delay-line policy

GR6J unit-hydrograph buffers and the HBV routing buffer remain fixed-size because
that preserves the published delay-line semantics and consistent batched tensor
shapes. Continuous parameters determine fixed-buffer ordinates through tensor
operations. Structural integers are non-calibrated module configuration.

## Calibration — dual stack

`ParamSpec` defines canonical names and bounds. `params_to_array`,
`array_to_parameters`, and `array_to_model` provide the model/flat-array adapter.

- `calibrate_gradient` uses a bounded sigmoid transform, explicit parameter
  mappings, `torch.optim`, and optional no-grad SSM warmup forcing. It returns a
  calibrated model and detached loss history.
- `calibrate_evolutionary` and `calibrate_nsga2` retain ctrl-freak's NumPy
  population/result boundary. Each complete population is represented by the
  leading batch dimension and evaluated in one Torch call under
  `torch.no_grad()`. They return `(model, GAResult)` and
  `(list[model], NSGA2Result)` respectively. Their integer `warmup` slices metric
  evaluation and is not SSM warmup forcing.

Metrics are Torch-native and differentiable.

## Precision and devices

Float64 on CPU is the reference and golden-fixture path. Float32 on an explicitly
selected accelerator is the training path; MPS does not provide float64, which is
one reason references remain on CPU. `reference_tensor`, `training_tensor`,
`reference_defaults`, and `training_defaults` make explicit local conversions.
Models preserve caller dtype/device. The package neither mutates process-global
defaults nor performs an import-time precision guard.

## Validation — equations are the specification

Version fidelity to published GR6J (airGR) and HBV-Light formulations has been
audited. Three divergences were corrected: GR6J UH and HBV MAXBAS convolution use
read-after-shift routing; GR6J `actual_exchange_total` includes the exponential
store exchange leg; and HBV routes above-field-capacity soil water to upper-zone
recharge. Two behaviors remain by decision: explicit-split forward-Euler store
over-draw, consistent with standard HBV implementations, and the internal
parameter-bound convention. GR6J constants and clamps match airGR. See
[GR6J](gr6j.md), [HBV](hbv.md), and the [contributor contract](contracts.md).

## HDX I/O

HDX is the canonical documented scalar data interface while the model core stays
format-agnostic. `from_hdx` returns NumPy forcing-dictionary leaves, streamflow,
statics, mask, and times. `.torch(dtype=..., device=...)` explicitly constructs a
model forcing dataclass and converts numerical leaves, including a boolean Torch
mask.

Single-basin dynamic NumPy leaves are `[T]`; multi-basin data use padded `[B,Tmax]`
tensors plus a mask. Model execution always consumes leading `[B,T]` forcing.
Optional HDX dependencies load lazily. The vocabulary layer maps role-opaque HDX
field names to hydrologeez forcing and target names, and predictions are written
as conformant scalar-dynamic HDX datasets.

### Future raster preprocessing

If a future model needs a DEM, preprocessing will reduce the raster to fixed
structural configuration and scalar attributes outside the differentiable model.
The model will not differentiate through raster I/O.

## Scope and sequence

Shipped scope is daily GR6J and single-zone HBV, both as Torch SSMs, with native
batch execution, Torch metrics, HDX I/O, gradient calibration, and evolutionary
calibration. GR2M, CemaNeige, glacier coupling, and multi-zone HBV remain out of
scope. The base contract remains open to SSM composition.

## Packaging and conventions

- `uv_build`, src layout, and Python 3.11 or newer.
- Torch as the numerical runtime; optional HDX Parquet dependencies.
- `uv` only for dependency and command execution.
- `ruff`, `ty`, and `pytest`; modern Python typing.
- Release-only version bumps through `bump-my-version`; publishing through GitHub
  Releases and Trusted Publishing/OIDC.
- Library-specific test assertions: `numpy.testing` for NumPy and
  `torch.testing.assert_close` for Torch, plus xarray/polars utilities.
- mkdocs-material, with `contracts.md` as contributor governance.

## Quality bar and open items

The quality floor is equation fidelity, regression tests, strict documentation,
continuous integration, executable examples, and clear contributor contracts.
Current concerns are maintaining finite, useful gradients near piecewise model
boundaries and evolving the hydrologeez-owned HDX vocabulary only when another
consumer establishes a shared profile.
