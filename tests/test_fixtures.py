"""Schema and non-vacuousness checks for the committed GR6J oracle fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).parent / "fixtures"

FLUX_KEYS = (
    "pet",
    "precip",
    "production_store",
    "net_rainfall",
    "storage_infiltration",
    "actual_et",
    "percolation",
    "effective_rainfall",
    "q9",
    "q1",
    "routing_store",
    "exchange",
    "actual_exchange_routing",
    "actual_exchange_direct",
    "actual_exchange_total",
    "qr",
    "qrexp",
    "exponential_store",
    "qd",
    "streamflow",
)


def _load(name: str) -> dict[str, np.ndarray]:
    with np.load(FIXTURES / name, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _check_run_fixture(name: str, expected_params: list[float]) -> dict[str, np.ndarray]:
    d = _load(name)
    for key in FLUX_KEYS:
        assert key in d, f"missing flux array {key} in {name}"
    n = d["streamflow"].size
    assert n == 12333
    for key in FLUX_KEYS:
        assert d[key].size == n, f"{key} length mismatch in {name}"
        assert np.all(np.isfinite(d[key])), f"non-finite values in {key} of {name}"
    np.testing.assert_array_equal(d["param_names"], ["x1", "x2", "x3", "x4", "x5", "x6"])
    np.testing.assert_allclose(d["params"], expected_params)
    assert int(d["warmup_length"]) == 365
    assert str(d["basin_id"]) == "camels_06224000"
    return d


def test_canonical_fixture():
    _check_run_fixture("gr6j_camels_06224000.npz", [350.0, 0.0, 90.0, 1.7, 0.0, 5.0])


def test_exchange_fixture_params_and_clamp():
    d = _check_run_fixture("gr6j_camels_06224000_exchange.npz", [350.0, 1.0, 90.0, 1.7, 0.5, 5.0])
    assert d["params"][1] == 1.0
    assert d["params"][4] == 0.5
    assert np.any(d["exchange"] != 0.0)
    # negative-R routing clamp fires (non-vacuous) ...
    assert np.any(d["actual_exchange_routing"] != d["exchange"])
    # ... and non-clamp steps also exist (both exchange branches present)
    assert np.any(d["actual_exchange_routing"] == d["exchange"])
    # Rust definition: total == routing + direct (excludes F)
    np.testing.assert_allclose(
        d["actual_exchange_total"],
        d["actual_exchange_routing"] + d["actual_exchange_direct"],
        atol=1e-12,
    )


def test_uh_ordinate_table():
    d = _load("gr6j_uh_ordinates.npz")
    x4 = d["x4_grid"]
    np.testing.assert_array_equal(x4, [0.5, 1.0, 1.7, 2.0, 3.5, 5.0, 7.0, 10.0])
    assert d["uh1_ord"].shape == (x4.size, 20)
    assert d["uh2_ord"].shape == (x4.size, 40)
    assert np.all(np.isfinite(d["uh1_ord"]))
    assert np.all(np.isfinite(d["uh2_ord"]))
    np.testing.assert_allclose(d["uh1_ord"].sum(axis=1), np.ones(x4.size), atol=1e-10)
    np.testing.assert_allclose(d["uh2_ord"].sum(axis=1), np.ones(x4.size), atol=1e-10)


def test_step_branches_positive_and_clamp():
    d = _load("gr6j_step_branches.npz")
    k = d["qrexp"].size
    assert d["input_states"].shape == (k, 63)
    assert d["params"].shape == (6,)
    assert d["precip"].shape == (k,)
    assert d["pet"].shape == (k,)
    assert d["uh1_ord"].shape == (20,)
    assert d["uh2_ord"].shape == (40,)
    assert d["exponential_store"].shape == (k,)
    for key in ("qrexp", "exponential_store", "input_states", "uh1_ord", "uh2_ord"):
        assert np.all(np.isfinite(d[key]))
    # AR = exp_store / x6 from the stored inputs (never re-typed)
    ar = d["input_states"][:, 2] / d["params"][5]
    np.testing.assert_allclose(ar, d["ar_target"])
    assert np.any(ar > 7.0)  # large-positive softplus branch exercised
    assert np.any(ar >= 33.0)  # +33 AR clamp exercised
