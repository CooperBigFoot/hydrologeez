# Hybrid HBV

Install `uv sync --locked --extra hybrid`. PyTorch is optional; existing NumPy/Numba HBV callers are unchanged.

Run both encoders and both training strategies with synthetic meteorology:

```sh
uv run --locked --extra hybrid python -m scripts.hybrid_synthetic
uv run --locked --extra hybrid python -m scripts.hybrid_synthetic --device mps
```

The example writes checkpoints to `.hybrid-example/`. No CAMELS data, NVIDIA GPU, FLA installation, or remote service is needed. See [local experiment evidence](../experiments/hybrid-hbv/README.md).

## Components and information context

`hydrologeez.hybrid.model.HybridHBV` combines a selectable one-layer LSTM or single-head Kimi Delta Attention encoder, a feed-forward parameter head, and a separate feed-forward reconstruction head. `ModelConfig` records the encoder, ordered meteorological features, ordered attributes, hidden width, and routing support. The default width is 16; the example uses 8. There is no shared optimization between the example's independently initialized strategies.

Both encoders are causal: each day's representation uses earlier inputs and visible same-day features, not later days. The parameter head uses the **last non-padding representation of the complete supplied context window**. Its fixed basin parameters therefore use later meteorology within that window when simulating earlier days of that same window. This is offline window parameter estimation, **not causal discharge forecasting**. A preceding context window can instead be supplied separately from the physical simulation period. Context is reset for every batch; no neural recurrent cache crosses windows.

The same preprocessing and representations serve pretraining and discharge learning. The head predicts one static parameter vector per basin, with zone parameters shared inside that basin. Elevation/vegetation zones retain distinct forcings and water stores. This baseline does not infer dynamic parameters or Feng-style parallel components.

## Inputs, masking, and losses

`runtime.Batch` contains:

- `meteorology`: `[basin, context day, feature]`; genuine missing observations may be NaN.
- `attributes`: `[basin, attribute]`, finite, in configured order. Supply already-scaled attributes; this baseline does not fit attribute scaling.
- `padding`: boolean `[basin, context day]`. Only right padding is supported; every sequence needs at least one non-padding day.
- `forcing`: finite physical `[simulation day, basin, zone, P/T/PET]`, in mm/day and Celsius, never normalized neural values.
- `area`: `[basin, zone]`, nonnegative fractions summing to one. Zero-area extra zones may represent unequal zone counts, but still require valid physical inputs and parameters.
- Optional observed `discharge` and boolean `discharge_valid`, both `[simulation day, basin]`; and explicit initial HBV `Memory`.

Feature order is a caller contract: tensors have no column labels. Check incoming data against `model.config.features` and `.attributes`. Physical P/T/PET order is fixed regardless of neural feature order.

Fit `data.Normalization.fit(values, padding, features)` explicitly on the authorized meteorological training pool. It ignores missing and padded observations, uses population standard deviation, and gives constant features scale one. It never fits inside a prediction or training step. Checkpoints store its mean, scale and feature order.

`data.prepare` supplies three channels per feature group: normalized visible values, visible-observation indicators, and synthetic-hidden indicators. Hidden values are zeroed only *after* fixed normalization, and never enter neural inputs. Genuine missing and padding values are not reconstruction targets. Legitimate normalized zeros remain distinguishable through indicators. The caller supplies a boolean synthetic mask; the example independently hides 25% of entries. Hidden valid targets alone contribute to normalized MSE. A mask with no targets raises an error. This is a starting scheme, not a scientific masking optimum.

Discharge loss defaults to MSE in physical discharge units on the caller's valid, finite targets. Select the differentiable NPE surrogate with `Training(model, discharge_loss="soft_npe", softness=0.1)`; it also applies after switching to the frozen stage, but never to reconstruction pretraining. See [shared discharge losses](losses.md) for dimensions, basin weighting, undefined-input policy and quadratic ranking cost. Exclude warm-up days and simulation padding with `discharge_valid`; context padding does not silently define simulation-period validity. Physical simulation cannot use missing forcings. Sampling, early stopping and regional evaluation belong in experiment assembly, not this runtime.

## Training and state

`runtime.Training` owns Adam, gradient checks, and the three explicit stages:

```python
training = Training(model, stage="direct", learning_rate=0.001)
training.train_step(batch)  # delta: encoder + parameter head
training.evaluate(validation_batch)  # no gradients or updates
prediction = training.predict(batch)  # no reconstruction head

pretraining = Training(other_model, stage="pretrain")
pretraining.train_step(batch, hidden=hidden_mask)
pretraining.save("pretrained.pt")
alpha = Training.load("pretrained.pt")
alpha.set_stage("frozen")  # resets optimizer moments
alpha.train_step(batch)  # only parameter head learns
alpha.save("discharge.pt")
```

The frozen encoder has `requires_grad=False` and stays in evaluation mode even when the parent model enters training mode. The pretext head is unused in discharge inference. Stage changes reset optimizer moments; saving/loading within a stage retains Adam state. Checkpoints include architecture, feature order, normalization, parameter bounds, weights, stage, optimizer, discharge loss and softness. Version 2 preserves the objective when resuming; version 1 loads with its original MSE objective. Unknown versions are rejected. Load uses PyTorch's weights-only reader. They do not package input datasets, physical forcing arrays, random-generator state, or HBV stores. A checkpoint reproduces inference for the same supplied inputs; it is not an automatic bitwise replay of future stochastic mask generation.

Physical memory is independent of KDA learned memory. `hbv.simulate` returns discharge and explicit final snow/liquid/soil, groundwater and routing stores. No input memory is mutated. Empty stores are the default. Warm-up is ordinary simulation, with its resulting memory passed forward. For restart, **keep the same estimated parameters**; the example computes parameters once, splits physical forcing, and passes `first.final` to the second call. Re-estimating parameters from another context is a new simulation, not a valid restart.

## Physical mapping and differentiability

The tensor implementation follows [existing daily HBV](hbv-light.md): snow partition/melt/refreeze/retention, beginning-of-day soil recharge and overflow, evaporation, percolation-first groundwater with nonlinear upper response, then integrated triangular routing. It supports dense basin/zone tensors rather than the NumPy simulator's packed simulation-to-basin mapping. Existing APIs remain unchanged.

The parameter head uses sigmoid mappings into these exploratory ranges:

| Parameter | Lower | Upper |
|---|---:|---:|
| TT | -2 | 2 |
| CFMAX | 0.1 | 5 |
| CFR | 0.01 | 0.2 |
| CWH | 0.01 | 0.2 |
| SCF | 0.5 | 1.5 |
| FC | 10 | 500 |
| BETA | 0.3 | 3 |
| LP | 0.1 | 1 |
| PERC | 0.01 | 4 |
| Alpha | 0.01 | 0.5 |
| K1 | 0.001 | 0.2 |
| K2 | 0.001 | 0.1 |
| MAXBAS | 1 | routing support (default 7 days) |

These are baseline search ranges, not calibrated or universal bounds. They are stored as checkpoint buffers. Direct tensor HBV accepts wider physically valid ranges. TT has no gradient through the precipitation partition comparison, though melt/refreeze provide other paths. Available-water limits and caps can yield zero gradients; threshold/tie derivatives are not uniquely defined. Near empty stores, exact forward powers are retained but the derivative at exactly zero is defined as zero to avoid `log(0)` in exponent derivatives. No smoothing changes the water laws. Gradient checks apply away from boundaries, not a promise that every parameter is identifiable in every window.

MAXBAS retains the exact integrated triangular CDF. A fixed seven-bin default support gives six future-arrival memory entries; entries beyond `ceil(MAXBAS)-1` are zero for compatible initial states. This removes parameter-dependent tensor lengths without truncating the chosen range. Duration gradients pass through CDF weights; bin edges and midpoint crossings remain piecewise boundaries. CPU float64 is used for physical parity/finite-difference tests; MPS uses float32.

## Scientific limits

Masked reconstruction quality does not establish useful frozen hydrological representations. These synthetic checks establish local execution and gradient paths, not improved regional accuracy, model identifiability, long-sequence speed or reproduction of Feng et al., Kimi K3 or a language model.

For regional studies, record the meteorological pool and time coverage used for fitting preprocessing and alpha pretraining. Unlabeled held-out-region meteorology is permitted only as explicitly described **transductive access**. Held-out discharge must never enter training or model selection; keep the supervised delta comparator clean. Large campaigns and regional evaluation belong in `uzh-results`; CAMELS storage/readers are separate work.
