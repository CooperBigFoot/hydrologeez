from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from conftest import write_scalar_hdx

from hydrologeez.hdx._polars import require_polars
from hydrologeez.hdx.loader import from_hdx
from hydrologeez.hdx.vocabulary import Vocabulary
from hydrologeez.models.gr6j import GR6JForcing
from hydrologeez.models.hbv import HBVForcing


def test_single_basin_gr6j(tmp_path: Path) -> None:
    root = _write_single_basin(tmp_path)

    data = from_hdx(root, forcing_type=GR6JForcing)

    assert isinstance(data.forcing, GR6JForcing)
    assert data.forcing.precip.shape == (3,)
    assert data.forcing.precip.dtype == jnp.float64
    assert data.forcing.pet.shape == (3,)
    assert data.forcing.pet.dtype == jnp.float64
    assert data.streamflow is not None
    assert data.streamflow.shape == (3,)
    assert data.streamflow.dtype == jnp.float64
    assert data.mask is None
    assert data.statics["drainage_area"].shape == ()
    assert data.statics["drainage_area"].dtype == jnp.float64
    assert data.basin_ids == ("0001",)
    assert isinstance(data.times, np.ndarray)
    assert data.times.dtype == np.dtype("datetime64[us]")
    assert len(data.times) == 3
    npt.assert_allclose(data.forcing.precip, np.array([1.0, 2.0, 3.0]))
    npt.assert_allclose(data.streamflow, np.array([4.0, 5.0, 6.0]))


def test_single_basin_hbv(tmp_path: Path) -> None:
    root = _write_single_basin(tmp_path)

    data = from_hdx(root, forcing_type=HBVForcing)

    assert isinstance(data.forcing, HBVForcing)
    assert data.forcing.precip.shape == (3,)
    assert data.forcing.precip.dtype == jnp.float64
    assert data.forcing.pet.shape == (3,)
    assert data.forcing.pet.dtype == jnp.float64
    assert data.forcing.temp.shape == (3,)
    assert data.forcing.temp.dtype == jnp.float64
    npt.assert_allclose(data.forcing.temp, np.array([7.0, 8.0, 9.0]))


def test_multi_basin_ragged_gr6j(synth_hdx_dataset: Path) -> None:
    data = from_hdx(synth_hdx_dataset, forcing_type=GR6JForcing)

    assert isinstance(data.forcing, GR6JForcing)
    assert data.forcing.precip.shape == (2, 6)
    assert data.forcing.precip.dtype == jnp.float64
    assert data.forcing.pet.shape == (2, 6)
    assert data.forcing.pet.dtype == jnp.float64
    assert data.mask is not None
    assert data.mask.shape == (2, 6)
    assert data.mask.dtype == jnp.bool_
    npt.assert_array_equal(
        data.mask, np.array([[True, True, True, True, True, True], [True, True, True, True, False, False]])
    )
    assert data.streamflow is not None
    assert data.streamflow.shape == (2, 6)
    assert data.streamflow.dtype == jnp.float64
    assert data.statics["drainage_area"].shape == (2,)
    assert data.statics["drainage_area"].dtype == jnp.float64
    assert data.basin_ids == ("0001", "0002")
    assert isinstance(data.times, tuple)
    assert [len(times) for times in data.times] == [6, 4]
    assert all(times.dtype == np.dtype("datetime64[us]") for times in data.times)
    npt.assert_allclose(data.forcing.precip[1, 4:], np.array([0.0, 0.0]))
    npt.assert_allclose(data.streamflow[1, 4:], np.array([0.0, 0.0]))


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

    assert isinstance(data.forcing, GR6JForcing)
    npt.assert_allclose(data.forcing.precip, np.array([1.0, 2.0, 3.0]))
    assert data.streamflow is not None
    npt.assert_allclose(data.streamflow, np.array([4.0, 5.0, 6.0]))


def test_forcing_none_when_required_columns_missing_or_not_requested(
    synth_streamflow_only_dataset: Path,
    synth_hdx_dataset: Path,
) -> None:
    streamflow_only = from_hdx(synth_streamflow_only_dataset, forcing_type=GR6JForcing)

    assert streamflow_only.forcing is None
    assert streamflow_only.streamflow is not None
    assert streamflow_only.streamflow.shape == (2, 6)
    assert streamflow_only.basin_ids == ("0001", "0002")
    assert isinstance(streamflow_only.times, tuple)
    assert streamflow_only.statics["drainage_area"].shape == (2,)

    not_requested = from_hdx(synth_hdx_dataset, forcing_type=None)

    assert not_requested.forcing is None
    assert not_requested.streamflow is not None
    assert not_requested.statics["drainage_area"].shape == (2,)


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
    assert data.streamflow is not None
    assert data.streamflow.shape == (2, 3)
    assert data.mask is not None
    npt.assert_array_equal(np.asarray(data.mask).sum(axis=1), np.array([3, 2]))


def test_hive_basin_column_is_ignored(synth_hdx_dataset: Path) -> None:
    pl = require_polars()
    hive_frame = pl.read_parquet(str(synth_hdx_dataset / "basin=*" / "scalar_dynamic.parquet"), hive_partitioning=True)
    assert "basin" in hive_frame.columns

    data = from_hdx(synth_hdx_dataset, forcing_type=GR6JForcing)

    assert data.basin_ids == ("0001", "0002")
    assert "basin" not in data.statics


def test_format_version(tmp_path: Path) -> None:
    unsupported = _write_single_basin(tmp_path / "unsupported", format_version="0.1")

    with pytest.raises(ValueError, match="0.1"):
        from_hdx(unsupported, forcing_type=GR6JForcing)

    supported = _write_single_basin(tmp_path / "supported", format_version="0.2")
    data = from_hdx(supported, forcing_type=GR6JForcing)
    assert data.basin_ids == ("0001",)


def test_geometry_less_dataset(geometry_less_root: Path) -> None:
    data = from_hdx(geometry_less_root, forcing_type=GR6JForcing)

    assert data.forcing is None
    assert data.streamflow is not None
    assert data.statics["drainage_area"].shape == (3,)
    assert data.basin_ids == ("0001", "0002", "0003")


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
