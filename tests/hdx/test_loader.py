from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
import torch
from conftest import write_scalar_hdx

from hydrologeez.hdx._polars import require_polars
from hydrologeez.hdx.loader import from_hdx
from hydrologeez.hdx.vocabulary import Vocabulary
from hydrologeez.models.gr6j import GR6JForcing
from hydrologeez.models.hbv import HBVForcing


def test_single_basin_gr6j(tmp_path: Path) -> None:
    data = from_hdx(_write_single_basin(tmp_path), forcing_type=GR6JForcing)

    assert isinstance(data.forcing, dict)
    assert not isinstance(data.forcing, GR6JForcing)
    for value in data.forcing.values():
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.float64
        assert value.shape == (3,)
    assert isinstance(data.streamflow, np.ndarray)
    assert data.streamflow.dtype == np.float64
    assert data.streamflow.shape == (3,)
    assert data.mask is None
    assert isinstance(data.statics["drainage_area"], np.ndarray)
    assert data.statics["drainage_area"].dtype == np.float64
    assert data.statics["drainage_area"].shape == ()
    assert data.basin_ids == ("0001",)
    assert isinstance(data.times, np.ndarray)
    assert data.times.dtype == np.dtype("datetime64[us]")
    npt.assert_allclose(data.forcing["precip"], np.array([1.0, 2.0, 3.0]))
    assert isinstance(data.streamflow, np.ndarray)
    npt.assert_allclose(np.asarray(data.streamflow, dtype=np.float64), np.array([4.0, 5.0, 6.0]))


def test_single_basin_hbv(tmp_path: Path) -> None:
    data = from_hdx(_write_single_basin(tmp_path), forcing_type=HBVForcing)

    assert isinstance(data.forcing, dict)
    assert list(data.forcing) == ["precip", "pet", "temp"]
    assert all(isinstance(value, np.ndarray) and value.dtype == np.float64 for value in data.forcing.values())
    npt.assert_allclose(data.forcing["temp"], np.array([7.0, 8.0, 9.0]))


def test_multi_basin_ragged_gr6j(synth_hdx_dataset: Path) -> None:
    data = from_hdx(synth_hdx_dataset, forcing_type=GR6JForcing)

    assert isinstance(data.forcing, dict)
    assert all(value.shape == (2, 6) and value.dtype == np.float64 for value in data.forcing.values())
    assert isinstance(data.mask, np.ndarray)
    assert data.mask.dtype == np.bool_
    npt.assert_array_equal(
        data.mask, np.array([[True, True, True, True, True, True], [True, True, True, True, False, False]])
    )
    assert isinstance(data.streamflow, np.ndarray)
    assert data.streamflow.shape == (2, 6)
    assert data.streamflow.dtype == np.float64
    assert data.statics["drainage_area"].shape == (2,)
    assert data.statics["drainage_area"].dtype == np.float64
    assert data.basin_ids == ("0001", "0002")
    assert isinstance(data.times, tuple)
    assert [len(times) for times in data.times] == [6, 4]
    assert all(times.dtype == np.dtype("datetime64[us]") for times in data.times)
    npt.assert_array_equal(data.forcing["precip"][1, 4:], np.zeros(2))
    npt.assert_array_equal(data.streamflow[1, 4:], np.zeros(2))


def test_float64_cpu_conversion_shares_storage(synth_hdx_dataset: Path) -> None:
    data = from_hdx(synth_hdx_dataset, forcing_type=GR6JForcing)
    converted = data.torch(dtype=torch.float64, device="cpu")

    assert isinstance(data.forcing, dict)
    assert isinstance(converted.forcing, GR6JForcing)
    floating = [converted.forcing.precip, converted.forcing.pet, converted.streamflow, *converted.statics.values()]
    assert all(isinstance(value, torch.Tensor) for value in floating)
    assert all(value is not None and value.dtype == torch.float64 and value.device.type == "cpu" for value in floating)
    assert converted.mask is not None
    assert converted.mask.dtype == torch.bool
    assert converted.mask.device.type == "cpu"
    torch.testing.assert_close(converted.forcing.precip, torch.as_tensor(data.forcing["precip"]))
    torch.testing.assert_close(converted.streamflow, torch.as_tensor(data.streamflow))
    torch.testing.assert_close(converted.statics["drainage_area"], torch.as_tensor(data.statics["drainage_area"]))
    torch.testing.assert_close(converted.mask, torch.as_tensor(data.mask))
    assert converted.forcing.precip.data_ptr() == data.forcing["precip"].__array_interface__["data"][0]
    assert isinstance(data.streamflow, np.ndarray)
    assert converted.times is data.times
    assert converted.basin_ids is data.basin_ids


@pytest.mark.parametrize(
    ("forcing_type", "expected_type", "keys"),
    [(GR6JForcing, GR6JForcing, ("precip", "pet")), (HBVForcing, HBVForcing, ("precip", "pet", "temp"))],
)
def test_conversion_constructs_requested_forcing(
    tmp_path: Path,
    forcing_type: type[GR6JForcing] | type[HBVForcing],
    expected_type: type[GR6JForcing] | type[HBVForcing],
    keys: tuple[str, ...],
) -> None:
    data = from_hdx(_write_single_basin(tmp_path), forcing_type=forcing_type)
    converted = data.torch(dtype=torch.float64, device="cpu")

    assert isinstance(data.forcing, dict)
    assert isinstance(converted.forcing, expected_type)
    for key in keys:
        torch.testing.assert_close(getattr(converted.forcing, key), torch.as_tensor(data.forcing[key]))


def test_float32_conversion_uses_selected_device(synth_hdx_dataset: Path) -> None:
    device = _available_device()
    data = from_hdx(synth_hdx_dataset, forcing_type=GR6JForcing)
    converted = data.torch(dtype=torch.float32, device=device)

    assert isinstance(data.forcing, dict)
    assert isinstance(converted.forcing, GR6JForcing)
    floating = [converted.forcing.precip, converted.forcing.pet, converted.streamflow, *converted.statics.values()]
    assert all(
        value is not None and value.dtype == torch.float32 and value.device.type == device.type for value in floating
    )
    assert converted.mask is not None
    assert converted.mask.dtype == torch.bool
    assert converted.mask.device.type == device.type
    torch.testing.assert_close(
        converted.forcing.precip.cpu(), torch.as_tensor(data.forcing["precip"], dtype=torch.float32)
    )


def test_vocabulary_override_resolves_foreign_columns(tmp_path: Path) -> None:
    root = write_scalar_hdx(
        tmp_path / "foreign",
        basins={"0001": 3},
        static_fields={"drainage_area": {"0001": 100.0}},
        dynamic_fields={
            "precip": {"0001": [1.0, 2.0, 3.0]},
            "pet": {"0001": [0.1, 0.2, 0.3]},
            "streamflow": {"0001": [4.0, 5.0, 6.0]},
        },
        column_names={"precip": "P", "pet": "PET", "streamflow": "Q"},
    )
    data = from_hdx(
        root,
        forcing_type=GR6JForcing,
        vocabulary=Vocabulary(overrides={"P": "precip", "PET": "pet", "Q": "streamflow"}),
    )
    assert isinstance(data.forcing, dict)
    npt.assert_allclose(data.forcing["precip"], np.array([1.0, 2.0, 3.0]))
    assert isinstance(data.streamflow, np.ndarray)
    npt.assert_allclose(np.asarray(data.streamflow, dtype=np.float64), np.array([4.0, 5.0, 6.0]))


def test_forcing_none_when_missing_or_not_requested(
    synth_streamflow_only_dataset: Path, synth_hdx_dataset: Path
) -> None:
    missing = from_hdx(synth_streamflow_only_dataset, forcing_type=GR6JForcing)
    assert missing.forcing is None
    assert isinstance(missing.streamflow, np.ndarray)
    converted = missing.torch(dtype=torch.float64, device="cpu")
    assert converted.forcing is None
    assert isinstance(converted.streamflow, torch.Tensor)

    not_requested = from_hdx(synth_hdx_dataset, forcing_type=None)
    assert not_requested.forcing is None
    assert not_requested.torch(dtype=torch.float64, device="cpu").forcing is None


def test_forcing_none_when_required_column_missing_from_one_basin(tmp_path: Path) -> None:
    root = write_scalar_hdx(
        tmp_path / "partial-forcing",
        basins={"0001": 3, "0002": 2},
        static_fields={"drainage_area": {"0001": 100.0, "0002": 250.0}},
        dynamic_fields={
            "precip": {"0001": [1.0, 2.0, 3.0], "0002": [10.0, 20.0]},
            "pet": {"0001": [0.1, 0.2, 0.3]},
            "streamflow": {"0001": [4.0, 5.0, 6.0], "0002": [40.0, 50.0]},
        },
    )
    data = from_hdx(root, forcing_type=GR6JForcing)
    assert data.forcing is None
    assert isinstance(data.mask, np.ndarray)
    npt.assert_array_equal(data.mask.sum(axis=1), np.array([3, 2]))


def test_absent_target_preserves_and_converts_forcing(tmp_path: Path) -> None:
    root = write_scalar_hdx(
        tmp_path / "forcing-only",
        basins={"0001": 2},
        static_fields={"drainage_area": {"0001": 100.0}},
        dynamic_fields={"precip": {"0001": [1.0, 2.0]}, "pet": {"0001": [0.1, 0.2]}},
    )
    data = from_hdx(root, forcing_type=GR6JForcing)
    converted = data.torch(dtype=torch.float64, device="cpu")
    assert data.streamflow is None
    assert converted.streamflow is None
    assert isinstance(converted.forcing, GR6JForcing)


def test_hive_basin_column_is_ignored(synth_hdx_dataset: Path) -> None:
    pl = require_polars()
    hive_frame = pl.read_parquet(str(synth_hdx_dataset / "basin=*" / "scalar_dynamic.parquet"), hive_partitioning=True)
    assert "basin" in hive_frame.columns
    data = from_hdx(synth_hdx_dataset, forcing_type=GR6JForcing)
    assert data.basin_ids == ("0001", "0002")
    assert "basin" not in data.statics


def test_format_version(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="0.1"):
        from_hdx(_write_single_basin(tmp_path / "unsupported", format_version="0.1"), forcing_type=GR6JForcing)
    assert from_hdx(_write_single_basin(tmp_path / "supported"), forcing_type=GR6JForcing).basin_ids == ("0001",)


def test_geometry_less_dataset(geometry_less_root: Path) -> None:
    data = from_hdx(geometry_less_root, forcing_type=GR6JForcing)
    assert data.forcing is None
    assert isinstance(data.streamflow, np.ndarray)
    assert data.statics["drainage_area"].shape == (3,)
    assert data.basin_ids == ("0001", "0002", "0003")


def _available_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        try:
            torch.ones(1, device="mps")
        except RuntimeError:
            pass
        else:
            return torch.device("mps")
    return torch.device("cpu")


def _write_single_basin(root: Path, *, format_version: str = "0.2") -> Path:
    return write_scalar_hdx(
        root,
        basins={"0001": 3},
        static_fields={"drainage_area": {"0001": 100.0}},
        dynamic_fields={
            "precip": {"0001": [1.0, 2.0, 3.0]},
            "pet": {"0001": [0.1, 0.2, 0.3]},
            "temp": {"0001": [7.0, 8.0, 9.0]},
            "streamflow": {"0001": [4.0, 5.0, 6.0]},
        },
        format_version=format_version,
    )
