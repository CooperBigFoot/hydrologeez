from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt

from hydrologeez.hdx.loader import from_hdx
from hydrologeez.hdx.writer import to_hdx


def test_public_hdx_imports_work() -> None:
    from hydrologeez import from_hdx, to_hdx

    assert callable(from_hdx)
    assert callable(to_hdx)


def test_single_basin_prediction_round_trips(tmp_path: Path) -> None:
    streamflow = _single_streamflow()
    times = _times(len(streamflow))
    root = tmp_path / "single"

    to_hdx(root, streamflow, times, ["0001"], statics={"drainage_area": np.array(123.0)})
    data = from_hdx(root, forcing_type=None)

    assert data.forcing is None
    assert data.streamflow is not None
    npt.assert_allclose(data.streamflow, streamflow)
    assert data.basin_ids == ("0001",)
    assert isinstance(data.times, np.ndarray)
    npt.assert_array_equal(data.times, times)
    npt.assert_allclose(data.statics["drainage_area"], np.array(123.0))


def test_batched_prediction_round_trips(tmp_path: Path) -> None:
    streamflow = _batched_streamflow()
    times = [_times(streamflow.shape[1]), _times(streamflow.shape[1])]
    root = tmp_path / "batched"

    to_hdx(
        root,
        streamflow,
        times,
        ["0001", "0002"],
        statics={"drainage_area": np.array([100.0, 250.0])},
    )
    data = from_hdx(root, forcing_type=None)

    assert data.forcing is None
    assert data.streamflow is not None
    npt.assert_allclose(data.streamflow, streamflow)
    assert data.basin_ids == ("0001", "0002")
    assert isinstance(data.times, tuple)
    npt.assert_array_equal(data.times[0], times[0])
    npt.assert_array_equal(data.times[1], times[1])
    npt.assert_allclose(data.statics["drainage_area"], np.array([100.0, 250.0]))


def _single_streamflow() -> np.ndarray:
    return np.random.default_rng(20260711).random(20, dtype=np.float64)


def _batched_streamflow() -> np.ndarray:
    return np.random.default_rng(20260712).random((2, 20), dtype=np.float64)


def _times(length: int) -> np.ndarray:
    start = np.datetime64("2001-01-01", "us")
    return start + np.arange(length).astype("timedelta64[D]")
