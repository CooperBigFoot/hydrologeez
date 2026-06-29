# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-29

### Added

- Initial public release.
- Differentiable conceptual hydrological models in JAX: GR6J and HBV.
- Forward rainfall-runoff simulation producing streamflow from precipitation and
  potential evapotranspiration forcing.
- Float64 precision enforced at import time (fail-fast); requires `JAX_ENABLE_X64=1`.
- Calibration: gradient-based optimization (optax) and evolutionary optimization
  (GA / NSGA-II via ctrl-freak).
- Hydrological metrics, a state-space model interface, and a default streamflow
  observation model.
- Typed public API.

[Unreleased]: https://github.com/CooperBigFoot/hydrologeez/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/CooperBigFoot/hydrologeez/releases/tag/v0.1.0
