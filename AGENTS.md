# Project Instructions

## 0. Project Overview

hydrologeez is a library of differentiable conceptual hydrological (rainfall-runoff)
models implemented in JAX. It currently ships two models — GR6J and HBV — exposed through
a shared state-space-model interface, and produces simulated streamflow from precipitation
and potential-evapotranspiration forcing.

Because the models are differentiable end-to-end, parameters can be calibrated either by
gradient-based optimization (optax) or by gradient-free evolutionary optimization
(GA / NSGA-II via the `ctrl-freak` dependency). The package also provides hydrological
metrics and a streamflow observation model.

Float64 precision is mandatory and enforced at import: importing hydrologeez calls
`enforce_float64()`, which raises `RuntimeError` unless JAX x64 is enabled. Always run with
`JAX_ENABLE_X64=1` (CI sets it as a job environment variable).

## 1. Python Environment

Use `uv` exclusively.

- Add dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync environment: `uv sync`
- Run commands: `uv run <command>`
- Run tests: `uv run pytest`

Do not use `pip`, `poetry`, `conda`, or `pip-tools` directly.

## 2. Code Style

Use `ruff` for formatting and linting, and `ty` for type checking.

```bash
uv run ruff format
uv run ruff check --fix
uv run ty check
```

If `ty` is not installed yet:

```bash
uv add --dev ty
```

Use modern Python typing syntax:

- Prefer built-in generics: `list[str]`, `dict[str, int]`, `tuple[str, ...]`.
- Prefer `|` unions: `str | None`.
- Avoid importing legacy aliases from `typing` such as `List`, `Dict`, `Tuple`, or `Optional`.
- Import from `typing` only when needed for features with no built-in equivalent, such as `Protocol`, `Literal`, or `NewType`.

## 3. Versioning and Releases

The project follows Semantic Versioning. The version is recorded in two files kept in
lockstep by `bump-my-version`: the `version` field of `pyproject.toml` and `__version__`
in `src/hydrologeez/__init__.py` (the `[tool.bumpversion]` config and its file table
manage both). Do NOT edit either version string by hand, and do NOT bump the version on
every commit.

Bump the version only when preparing a release:

```bash
uv run bump-my-version bump patch   # or: minor, major
```

Bump `minor` or `major` only when explicitly requested; otherwise `patch`. In the same
release commit, update `CHANGELOG.md`: move the `[Unreleased]` notes into a new versioned
section with the release date, and refresh the compare/release link references.

Releases are published from GitHub, not from a developer machine. Cutting a non-prerelease
GitHub Release `vX.Y.Z` triggers `.github/workflows/release.yml`, which runs `uv build` and
`uv publish` to PyPI via Trusted Publishing (OIDC, no stored tokens). A prerelease GitHub
Release, or a manual `workflow_dispatch` with `target=testpypi`, publishes to TestPyPI
instead. Do not run `uv publish` or `twine` locally, and do not create release tags by
hand — the GitHub Release creates the tag.

## 4. Testing Complex Data Objects

Prefer third-party testing utilities over manual element-wise assertions when comparing complex data objects.

Avoid manually checking lengths, schemas, coordinates, dimensions, shapes, dtypes, or element-wise equality when a library-specific assertion exists.

### NumPy

Use `numpy.testing`.

```python
import numpy as np

np.testing.assert_array_equal(result, expected)
np.testing.assert_allclose(result, expected)
```

### Xarray

Use `xarray.testing`.

```python
import xarray as xr

xr.testing.assert_equal(result, expected)
xr.testing.assert_identical(result, expected)
xr.testing.assert_allclose(result, expected)
```

### Polars

Use `polars.testing`.

```python
import polars.testing as pl_testing

pl_testing.assert_frame_equal(result_df, expected_df)
pl_testing.assert_series_equal(result_series, expected_series)
```
