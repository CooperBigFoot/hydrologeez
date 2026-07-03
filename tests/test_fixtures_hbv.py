"""Schema and non-vacuousness checks for the committed HBV-Light oracle fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).parent / "fixtures"

FLUX_KEYS = (
    "precip",
    "temp",
    "pet",
    "precip_rain",
    "precip_snow",
    "snow_pack",
    "snow_melt",
    "liquid_water_in_snow",
    "snow_input",
    "soil_moisture",
    "recharge",
    "actual_et",
    "upper_zone",
    "lower_zone",
    "q0",
    "q1",
    "q2",
    "percolation",
    "qgw",
    "streamflow",
)

PARAM_NAMES = (
    "tt",
    "cfmax",
    "sfcf",
    "cwh",
    "cfr",
    "fc",
    "lp",
    "beta",
    "k0",
    "k1",
    "k2",
    "perc",
    "uzl",
    "maxbas",
)

CANONICAL_PARAMS = [0.0, 3.5, 1.0, 0.1, 0.05, 250.0, 0.7, 2.0, 0.3, 0.1, 0.05, 2.0, 20.0, 3.0]
MAXBAS25_PARAMS = [0.5, 5.0, 1.1, 0.1, 0.05, 300.0, 0.7, 2.5, 0.4, 0.15, 0.04, 2.5, 25.0, 2.5]
OVERFLOW_PARAMS = [0.0, 3.5, 1.0, 0.1, 0.05, 50.0, 0.7, 6.0, 0.3, 0.1, 0.05, 2.0, 20.0, 3.0]

MAXBAS_GRID = (1.0, 2.0, 2.5, 3.0, 3.5, 5.0, 7.0)


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
    np.testing.assert_array_equal(d["param_names"], list(PARAM_NAMES))
    np.testing.assert_allclose(d["params"], expected_params)
    assert int(d["warmup_length"]) == 365
    assert str(d["basin_id"]) == "camels_06224000"
    return d


def test_canonical_fixture():
    d = _check_run_fixture("hbv_camels_06224000.npz", CANONICAL_PARAMS)
    assert d["params"][13] == 3.0  # integer maxbas
    # same-day routing (read-after-shift): streamflow turns on the SAME step qgw
    # first does; the buggy read-before-shift delayed it one step.
    first_qgw = int(np.argmax(d["qgw"] > 0.0))
    assert first_qgw > 0
    assert int(np.argmax(d["streamflow"] > 0.0)) == first_qgw  # was first_qgw + 1


def test_maxbas25_fixture_params_and_snow():
    d = _check_run_fixture("hbv_camels_06224000_maxbas25.npz", MAXBAS25_PARAMS)
    assert d["params"][13] == 2.5  # fractional maxbas, exactly 2.5
    # cold-tail temps drive the snow routine non-vacuously
    assert np.count_nonzero(d["precip_snow"]) > 1
    assert np.count_nonzero(d["snow_melt"]) > 1


def test_overflow_fixture_fires_and_conserves_mass():
    d = _check_run_fixture("hbv_camels_06224000_overflow.npz", list(OVERFLOW_PARAMS))
    fc = float(d["params"][5])
    assert d["params"][5] == 50.0 and d["params"][7] == 6.0  # low fc / steep beta
    sm = d["soil_moisture"]
    # overflow FIRES: soil moisture is driven to field capacity on >= 1 step
    assert int(np.sum(sm >= fc - 1e-9)) >= 1
    # per-step soil mass balance with overflow accounted, NO discarded water:
    # snow_input == dSM + recharge_total + actual_et   (recharge column == base + overflow)
    sm_prev = np.concatenate([[0.5 * fc], sm[:-1]])
    resid = d["snow_input"] - ((sm - sm_prev) + d["recharge"] + d["actual_et"])
    assert np.max(np.abs(resid)) < 1e-9


def test_triangular_weights_table():
    d = _load("hbv_triangular_weights.npz")
    np.testing.assert_array_equal(d["maxbas_grid"], list(MAXBAS_GRID))
    weights = d["weights"]
    assert weights.shape == (len(MAXBAS_GRID), 7)
    assert np.all(np.isfinite(weights))
    # each kernel row sums to ~1 (normalize-by-sum applied in the oracle)
    np.testing.assert_allclose(weights.sum(axis=1), np.ones(len(MAXBAS_GRID)), atol=1e-6)
    # zero beyond ceil(maxbas); strictly positive within the active support
    for i, maxbas in enumerate(MAXBAS_GRID):
        active = int(np.ceil(maxbas))
        np.testing.assert_array_equal(weights[i, active:], np.zeros(7 - active))
        assert np.all(weights[i, :active] > 0.0)
