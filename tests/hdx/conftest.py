from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from hydrologeez.hdx._polars import require_polars


def write_scalar_hdx(
    root: Path,
    *,
    basins: Mapping[str, int],
    dynamic_fields: Mapping[str, Mapping[str, Sequence[float] | None]],
    static_fields: Mapping[str, Mapping[str, float]] | None = None,
    format_version: str = "0.2",
    column_names: Mapping[str, str] | None = None,
) -> Path:
    pl = require_polars()
    column_names = {} if column_names is None else column_names
    static_fields = (
        {"drainage_area": {basin_id: float(index + 1) * 100.0 for index, basin_id in enumerate(basins)}}
        if static_fields is None
        else static_fields
    )

    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": format_version,
        "name": root.name,
        "created_at": "2026-01-01T00:00:00Z",
        "producer_version": "hydrologeez-tests/0.0.0",
        "crs": "EPSG:4326",
        "cadence": "daily",
    }
    (root / "manifest.json").write_text(json.dumps(manifest))

    static_data = {"basin_id": list(basins)}
    for field_name, values_by_basin in static_fields.items():
        static_data[column_names.get(field_name, field_name)] = [values_by_basin[basin_id] for basin_id in basins]
    pl.DataFrame(static_data).write_parquet(root / "scalar_static.parquet", statistics=True)

    for basin_id, length in basins.items():
        basin_dir = root / f"basin={basin_id}"
        basin_dir.mkdir()
        dynamic_data = {
            "basin_id": [basin_id] * length,
            "time": [datetime(2000, 1, 1) + timedelta(days=day) for day in range(length)],
        }
        for field_name, values_by_basin in dynamic_fields.items():
            values = values_by_basin.get(basin_id)
            if values is None:
                continue
            dynamic_data[column_names.get(field_name, field_name)] = list(values)
        frame = pl.DataFrame(dynamic_data).with_columns(pl.col("time").cast(pl.Datetime("us")))
        frame.write_parquet(basin_dir / "scalar_dynamic.parquet", statistics=True)

    return root


@pytest.fixture
def synth_hdx_dataset(tmp_path: Path) -> Path:
    return write_scalar_hdx(
        tmp_path / "full",
        basins={"0001": 6, "0002": 4},
        static_fields={"drainage_area": {"0001": 100.0, "0002": 250.0}},
        dynamic_fields={
            "precip": {"0001": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "0002": [10.0, 20.0, 30.0, 40.0]},
            "pet": {"0001": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "0002": [1.0, 2.0, 3.0, 4.0]},
            "temp": {"0001": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0], "0002": [17.0, 18.0, 19.0, 20.0]},
            "streamflow": {"0001": [2.0, 3.0, 5.0, 7.0, 11.0, 13.0], "0002": [4.0, 6.0, 8.0, 10.0]},
        },
    )


@pytest.fixture
def synth_streamflow_only_dataset(tmp_path: Path) -> Path:
    return write_scalar_hdx(
        tmp_path / "streamflow-only",
        basins={"0001": 6, "0002": 4},
        static_fields={"drainage_area": {"0001": 100.0, "0002": 250.0}},
        dynamic_fields={
            "streamflow": {"0001": [2.0, 3.0, 5.0, 7.0, 11.0, 13.0], "0002": [4.0, 6.0, 8.0, 10.0]},
        },
    )


@pytest.fixture
def geometry_less_root(tmp_path: Path) -> Path:
    return write_scalar_hdx(
        tmp_path / "geometry-less",
        basins={"0001": 3, "0002": 3, "0003": 3},
        static_fields={"drainage_area": {"0001": 100.0, "0002": 250.0, "0003": 400.0}},
        dynamic_fields={
            "streamflow": {"0001": [1.0, 2.0, 3.0], "0002": [4.0, 5.0, 6.0], "0003": [7.0, 8.0, 9.0]},
        },
    )
