"""Independent preparation and ordering checks for the local experiment."""

from pathlib import Path

import numpy as np
import pytest

from scripts.hbv_input_path import oudin, prepare


def test_oudin_equatorial_equinox() -> None:
    # FAO extraterrestrial radiation near equinox: about 37.8 MJ/m2/day.
    result = oudin(np.array([20.0, -6.0]), np.array([80, 80]), 0.0)
    assert result[0] == pytest.approx(3.86, abs=0.02)
    assert result[1] == 0.0
    assert oudin(np.array([20.0]), np.array([246]), -20.0)[0] == pytest.approx(3.286, abs=0.01)


@pytest.mark.parametrize(
    "kind", ["numpy-time", "numpy-basin", "zarr-time", "zarr-basin", "parquet-files", "parquet-consolidated"]
)
@pytest.mark.parametrize("order", ["time", "basin"])
def test_reader_preserves_requested_identity_dates_and_values(tmp_path: Path, kind: str, order: str) -> None:
    from scripts.hbv_input_path import construct, load

    values = np.arange(36, dtype=np.float64).reshape(4, 3, 3)
    construct(tmp_path, ["001", "002", "003"], np.arange(4, dtype=np.int64), values)
    actual, _ = load(tmp_path, kind, ["003", "001"], 1, 3, order)
    np.testing.assert_array_equal(actual, values[1:3][:, [2, 0]])
    assert actual.dtype == np.float64
    assert actual.strides[0] == (48 if order == "time" else 24)


def test_prepare_rejects_missing_and_misaligned_days() -> None:
    rows = np.zeros((2, 11))
    rows[:, :3] = [[2000, 1, 1], [2000, 1, 3]]
    with pytest.raises(ValueError, match="daily"):
        prepare(rows, 45.0)
    rows[1, 2] = 2
    rows[0, 5] = -999
    with pytest.raises(ValueError, match="missing"):
        prepare(rows, 45.0)


def test_resident_memory_orders_preserve_complete_hbv_result() -> None:
    from hydrologeez.hbv_light import simulate
    from scripts.hbv_input_path import inputs

    rng = np.random.default_rng(904)
    values = rng.uniform(0, 8, (80, 3, 3))
    values[:, :, 1] -= 4
    contiguous = simulate(*inputs(values, 3))
    basin_major = np.ascontiguousarray(values.transpose(1, 0, 2)).transpose(1, 0, 2)
    reordered = simulate(*inputs(basin_major, 3))
    np.testing.assert_array_equal(reordered.outlet, contiguous.outlet)
    for name in ("zone", "groundwater", "routing"):
        np.testing.assert_array_equal(getattr(reordered.final, name), getattr(contiguous.final, name))


def test_source_columns_and_mean_temperature() -> None:
    # CAMELS 03049800, 1980-01-01: P=0, Tmax=.74, Tmin=-5.50.
    rows = np.array([[1980, 1, 1, 12, 32832, 0, 170.60, 0, 0.74, -5.50, 400.0]])
    dates, values = prepare(rows, 40.48)
    assert dates.tolist() == [722815]
    np.testing.assert_allclose(values[0, :2], [0, -2.38], rtol=0, atol=1e-14)
