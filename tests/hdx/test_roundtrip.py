from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt
import torch

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
    static = np.array(123.0)

    to_hdx(root, streamflow, times, ["0001"], statics={"drainage_area": static})
    data = from_hdx(root, forcing_type=None)

    assert data.forcing is None
    assert isinstance(data.streamflow, np.ndarray)
    assert data.streamflow.dtype == np.float64
    assert isinstance(data.statics["drainage_area"], np.ndarray)
    assert data.statics["drainage_area"].dtype == np.float64
    npt.assert_allclose(data.streamflow, streamflow)
    npt.assert_allclose(data.statics["drainage_area"], static)
    assert data.basin_ids == ("0001",)
    assert isinstance(data.times, np.ndarray)
    npt.assert_array_equal(data.times, times)

    converted = data.torch(dtype=torch.float64, device="cpu")
    assert converted.forcing is None
    assert converted.mask is None
    torch.testing.assert_close(converted.streamflow, torch.as_tensor(streamflow))
    torch.testing.assert_close(converted.statics["drainage_area"], torch.as_tensor(static))
    assert converted.times is data.times


def test_batched_prediction_round_trips(tmp_path: Path) -> None:
    streamflow = _batched_streamflow()
    times = [_times(streamflow.shape[1]), _times(streamflow.shape[1])]
    root = tmp_path / "batched"
    statics = np.array([100.0, 250.0])

    to_hdx(root, streamflow, times, ["0001", "0002"], statics={"drainage_area": statics})
    data = from_hdx(root, forcing_type=None)

    assert data.forcing is None
    assert isinstance(data.streamflow, np.ndarray)
    assert data.streamflow.dtype == np.float64
    assert isinstance(data.statics["drainage_area"], np.ndarray)
    assert data.statics["drainage_area"].dtype == np.float64
    npt.assert_allclose(data.streamflow, streamflow)
    npt.assert_allclose(data.statics["drainage_area"], statics)
    assert data.basin_ids == ("0001", "0002")
    assert isinstance(data.times, tuple)
    npt.assert_array_equal(data.times[0], times[0])
    npt.assert_array_equal(data.times[1], times[1])

    converted = data.torch(dtype=torch.float64, device="cpu")
    assert converted.forcing is None
    torch.testing.assert_close(converted.streamflow, torch.as_tensor(streamflow))
    torch.testing.assert_close(converted.statics["drainage_area"], torch.as_tensor(statics))
    assert converted.mask is not None
    assert converted.mask.dtype == torch.bool
    torch.testing.assert_close(converted.mask, torch.ones_like(converted.mask, dtype=torch.bool))
    assert converted.times is data.times


def _single_streamflow() -> np.ndarray:
    return np.random.default_rng(20260711).random(20, dtype=np.float64)


def _batched_streamflow() -> np.ndarray:
    return np.random.default_rng(20260712).random((2, 20), dtype=np.float64)


def _times(length: int) -> np.ndarray:
    start = np.datetime64("2001-01-01", "us")
    return start + np.arange(length).astype("timedelta64[D]")
