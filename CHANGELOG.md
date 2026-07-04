# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- Legacy Rust-port parity fixtures (`tests/fixtures/*.npz`) and their README, the
  13 fixture-touching test files and the docs-example test
  (`tests/hdx/test_example.py`), and both fixture-generator scripts
  (`scripts/generate_gr6j_fixtures.py`, `scripts/generate_hbv_fixtures.py`).
- The runnable quickstart examples (`docs/examples/`) and the docs-exec machinery
  that embedded and executed them.
- The README "Validated numerics" claim and the parity acceptance-bar section of
  `docs/contracts.md`; external-validation claims removed across the docs and the
  source docstrings.

### Changed

- Docs now describe only current package behavior: no external-validation or parity
  claims; constants and parameter bounds are stated bare, with no provenance
  citations to the retired Rust codebase.
- `docs/hdx.md` refreshed: basin-first on-disk layout tree taken verbatim from the
  HDX spec, links to the canonical HDX repo and spec, and the spec's
  shape-not-provenance framing.
- README gains a short note on the package name (hydrology + "geez").

### Fixed

- HDX test-suite conformance: the test helper now writes the exact six-field
  HDX 0.2 manifest floor, and a hardcoded absolute fixture path was replaced by a
  synthesized geometry-less dataset so the geometry-less loader test always runs
  instead of being skipped.

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
