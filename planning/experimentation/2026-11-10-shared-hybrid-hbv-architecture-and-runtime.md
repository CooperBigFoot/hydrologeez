# Shared hybrid HBV implementation decisions

Implements the [published vision](../visions/2026-11-10-shared-hybrid-hbv-architecture-and-runtime.md).

- Selected static basin parameters from the last valid full-context representation, with shared zone parameters and separate physical zone stores. Same-window estimates are offline, not causal discharge forecasts.
- Selected a one-layer LSTM and single-head, no-short-convolution KDA adapted from pinned FLA source; optimized FLA installation does not provide a verified Mac path.
- Retained hard HBV laws and exact triangular routing using fixed support; defined zero-store power derivatives as zero rather than smoothing forward physics.
- Used explicit normalization/masking, masked reconstruction MSE, discharge MSE, Adam, and three small runtime stages. Frozen encoders preserve evaluation behavior and weights.
- Chose optional PyTorch and synthetic CPU/MPS validation; no CAMELS reader, NVIDIA execution, training framework, or regional-performance claim.

[Measurements, pinned provenance, test evidence and limitations](../../experiments/hybrid-hbv/README.md) record results once. [Usage](../../docs/hybrid-hbv.md) specifies input context, parameter ranges, state, checkpoints and experimental information boundaries. Related implementation: [PR #176](https://github.com/CooperBigFoot/hydrologeez/pull/176).
