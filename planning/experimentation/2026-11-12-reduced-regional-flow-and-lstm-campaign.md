# Reduced regional flow and LSTM campaign: experimentation notes

Vision: [Reduced regional flow and LSTM campaign](../visions/2026-11-12-reduced-regional-flow-and-lstm-campaign.md)

## Standalone DischargeLSTM readout under transformed-discharge MSE

Question: When training `DischargeLSTM` with valid-element-weighted MSE in standardized `log1p(Q)` space (`Log(offset=1.0, units="mm/day")` followed by global `ZScore`), how should the LSTM readout head be parameterized so that predictions remain nonnegative in physical `mm/day` without silent clipping or double transformation?

Approaches tried on 16 diverse CAMELS training basins (`1,418` zero-flow days), 9 validation basins (`1,576` zero-flow days, `1989-10-01..1993-09-30`), and 13 held-out Region 1 basins (`1999-10-01..2005-09-30`, 80 Adam updates at `lr = 1e-3`):

- `transformed_linear`: unconstrained linear readout directly in standardized `log1p(Q)` space, inverted to physical discharge via `expm1(z * scale + mean)`.
- `expm1_softplus`: linear readout passed through `softplus` in `log1p(Q)` space, inverted to physical discharge via `expm1(softplus(raw))`.
- `softplus_physical`: `DischargeLSTM` with `output_activation="softplus"` emitting physical discharge `Q_pred = softplus(raw) > 0` in `mm/day`, with the fitted `Transform` (`log1p` then global `ZScore`) applied differentiably inside `Training.loss` (matching `HybridHBV`).

Observations:

- `transformed_linear` achieved `0.2622` validation transformed MSE, but predicted values below `-mean / scale` on dry days, inverting to negative physical discharge (`min Q = -0.096 mm/day` on `1,474` validation basin-days) and causing `metrics.evaluate` (`kge_np`) to fail without post-hoc clipping.
- `expm1_softplus` guaranteed positive physical discharge (`0` negative predictions, `0.3065` validation transformed MSE, `0.481` validation median NPE), but because physical discharge grows exponentially with positive readout activations (`exp(softplus(raw)) - 1`), out-of-sample meteorological spikes on held-out `1999–2005` basins caused peak overshoot (`max Q = 22.89 mm/day`) and reduced held-out median NPE to `0.602`.
- `softplus_physical` guaranteed strictly positive physical discharge (`0` negative predictions, `0.3091` validation transformed MSE, `0.459` validation median NPE) and generalized best on held-out `1999–2005` basins (`0.689` median NPE), because physical discharge grows linearly for positive readout activations while `discharge_transform.forward_tensor` maps physical `mm/day` predictions into standardized `log1p(Q)` space inside the loss.

Decision: keep `DischargeLSTM` in physical `mm/day` space with `output_activation="softplus"` and apply `discharge_transform.forward_tensor` inside `Training.loss` for both `HybridHBV` and `DischargeLSTM`. Persist `discharge_transform` in both runtimes' checkpoints and reject `output_activation="linear"` when `discharge_transform` is supplied to `lstm.runtime.Training`.
