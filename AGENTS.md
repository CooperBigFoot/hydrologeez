# Project Instructions

## 0. Project Overview

hydrologeez provides differentiable conceptual rainfall-runoff models as PyTorch
`nn.Module` state-space models. GR6J and HBV consume leading-batch forcing tensors,
run eager recurrences, and support Torch autograd. Gradient calibration uses
`torch.optim`; evolutionary GA/NSGA-II calibration uses ctrl-freak while evaluating
each population as one batched no-grad Torch call.

Dtype and device selection is explicit. Float64 on CPU is the numerical reference and
reproduction path; float32 is the accelerator training path. Code must preserve caller
dtype/device and must not change Torch global defaults.

## 1. Python Environment

Use `uv` exclusively.

- Add dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync environment: `uv sync`
- Run commands: `uv run <command>`
- Run tests: `uv run pytest`

Do not use `pip`, `poetry`, `conda`, or `pip-tools` directly.

## 2. Code Style and Acceptance Gates

Use `ruff` for formatting and linting, and `ty` for type checking. Run the canonical
non-mutating local gates:

```bash
uv run ruff format --check
uv run ruff check
uv run ty check
uv run pytest
```

If `ty` is not installed yet, use `uv add --dev ty`.

Use modern Python typing syntax:

- Prefer built-in generics: `list[str]`, `dict[str, int]`, `tuple[str, ...]`.
- Prefer `|` unions: `str | None`.
- Avoid legacy aliases such as `List`, `Dict`, `Tuple`, and `Optional`.
- Import from `typing` only for features without built-in equivalents, such as
  `Protocol`, `Literal`, or `NewType`.

## 3. Documentation Governance

Public examples must execute against real signatures. Forcing examples must show
`[B, T]` tensors. Build documentation with:

```bash
NO_MKDOCS_2_WARNING=true uv run --group docs mkdocs build --strict
```

The `hydrologeez.hcx` package is a development-only conformance adapter: it imports hcx
lazily and hydrologeez must publish no `hcx.models` entry point.

## 4. Versioning and Releases

The project follows Semantic Versioning. The `pyproject.toml` version and
`src/hydrologeez/__init__.py` `__version__` remain in lockstep through
`bump-my-version`; never edit either by hand and never bump for an ordinary commit.

Bump only while preparing a release with `uv run bump-my-version bump patch` (or
`minor`/`major` only when explicitly requested). The same release commit updates
`CHANGELOG.md`, moving Unreleased notes into a dated version section and refreshing
links.

Publishing occurs only through GitHub Releases and Trusted Publishing/OIDC. A normal
release publishes to PyPI; prereleases or the TestPyPI workflow input publish to
TestPyPI. Do not publish locally, run `twine`, create tags by hand, or trigger a release
for an ordinary change.

## 5. Testing Complex Data Objects

Prefer library testing utilities over manual checks of lengths, schemas, coordinates,
dimensions, shapes, dtypes, or elements.

### NumPy

Use `numpy.testing.assert_array_equal` and `numpy.testing.assert_allclose`.

### Torch

Use `torch.testing.assert_close`.

### Xarray

Use `xarray.testing.assert_equal`, `assert_identical`, or `assert_allclose`.

### Polars

Use `polars.testing.assert_frame_equal` and `assert_series_equal`.

<!-- BEGIN SYNCED DOCTRINE; source-sha256=59e37fd6b3dbab27530822e6956da51bb7ae76b637e3638530f99a8b4db9038d -->
Four rules. They are one design stance seen four ways: a module means one thing, receives exactly what it needs, in types that cannot lie, and dies rather than guess.

1. **A module means one thing.**
2. **It receives exactly what it needs.**
3. **Its types cannot lie.**
4. **It dies rather than guess.**
<!-- END SYNCED DOCTRINE -->
