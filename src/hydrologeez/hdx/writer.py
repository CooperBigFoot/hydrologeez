from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from hydrologeez.hdx._polars import require_pyarrow


def to_hdx(
    root,
    streamflow,
    times,
    basin_ids: Sequence[str],
    statics: Mapping[str, object] | None = None,
    *,
    field: str = "streamflow",
    name: str = "hydrologeez-prediction",
    crs: str = "EPSG:4326",
    cadence: str | None = None,
    producer_version: str | None = None,
) -> Path:
    pa = require_pyarrow()
    import pyarrow.parquet as pq

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    streamflow_array = np.asarray(streamflow, dtype=np.float64)
    basin_id_tuple = tuple(str(basin_id) for basin_id in basin_ids)
    per_basin_times = _normalize_times(times, streamflow_array, len(basin_id_tuple))

    for basin_index, basin_id in enumerate(basin_id_tuple):
        basin_times = per_basin_times[basin_index]
        values = _streamflow_for_basin(streamflow_array, basin_index, len(basin_times))
        basin_dir = root_path / f"basin={basin_id}"
        basin_dir.mkdir(parents=True, exist_ok=True)
        _write_dynamic(pa, pq, basin_dir / "scalar_dynamic.parquet", basin_id, basin_times, field, values)

    _write_static(pa, pq, root_path, basin_id_tuple, statics)
    _write_manifest(
        root_path,
        name=name,
        crs=crs,
        cadence=cadence if cadence is not None else _derive_cadence(per_basin_times),
        producer_version=producer_version,
    )
    return root_path


def _normalize_times(times: object, streamflow: np.ndarray, basin_count: int) -> tuple[np.ndarray, ...]:
    if streamflow.ndim == 1:
        if basin_count != 1:
            raise ValueError("1-D streamflow requires exactly one basin_id")
        return (_as_datetime_us(times),)
    if streamflow.ndim == 2:
        if streamflow.shape[0] != basin_count:
            raise ValueError("2-D streamflow leading dimension must match basin_ids")
        if not isinstance(times, Sequence) or isinstance(times, np.ndarray):
            raise ValueError("2-D streamflow requires a sequence of per-basin time arrays")
        if len(times) != basin_count:
            raise ValueError("times length must match basin_ids")
        return tuple(_as_datetime_us(basin_times) for basin_times in times)
    raise ValueError("streamflow must be 1-D [T] or 2-D [B, T]")


def _write_dynamic(
    pa,
    pq,
    path: Path,
    basin_id: str,
    times: np.ndarray,
    field: str,
    values: np.ndarray,
) -> None:
    order = np.argsort(times)
    sorted_times = times[order]
    sorted_values = values[order]
    schema = pa.schema(
        [
            pa.field("basin_id", pa.string(), nullable=False),
            pa.field("time", pa.timestamp("us"), nullable=False),
            pa.field(field, pa.float64(), nullable=True),
        ]
    )
    table = pa.table(
        {
            "basin_id": pa.array([basin_id] * len(sorted_times), type=pa.string()),
            "time": pa.array(sorted_times, type=pa.timestamp("us")),
            field: pa.array(sorted_values, type=pa.float64()),
        },
        schema=schema,
    )
    time_index = table.schema.get_field_index("time")
    pq.write_table(
        table,
        path,
        compression="zstd",
        write_statistics=True,
        sorting_columns=[pq.SortingColumn(column_index=time_index, descending=False, nulls_first=False)],
    )


def _as_datetime_us(times: object) -> np.ndarray:
    array = np.asarray(times, dtype="datetime64[us]")
    if array.ndim != 1:
        raise ValueError("times must be 1-D datetime64[us] arrays")
    return array


def _streamflow_for_basin(streamflow: np.ndarray, basin_index: int, length: int) -> np.ndarray:
    values = streamflow[:length] if streamflow.ndim == 1 else streamflow[basin_index, :length]
    if len(values) != length:
        raise ValueError("streamflow length is shorter than the corresponding time axis")
    return np.asarray(values, dtype=np.float64)


def _write_static(
    pa,
    pq,
    root: Path,
    basin_ids: tuple[str, ...],
    statics: Mapping[str, object] | None,
) -> None:
    fields = [pa.field("basin_id", pa.string(), nullable=False)]
    columns = {"basin_id": pa.array(list(basin_ids), type=pa.string())}
    if statics:
        for field_name, values in statics.items():
            fields.append(pa.field(field_name, pa.float64(), nullable=True))
            columns[field_name] = pa.array(_static_values(values, len(basin_ids)), type=pa.float64())
    pq.write_table(pa.table(columns, schema=pa.schema(fields)), root / "scalar_static.parquet", write_statistics=True)


def _static_values(values: object, basin_count: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if basin_count == 1:
        if array.ndim == 0:
            return np.asarray([float(array)], dtype=np.float64)
        if array.shape == (1,):
            return np.asarray(array, dtype=np.float64)
        raise ValueError("single-basin static values must be scalar or length 1")
    if array.shape != (basin_count,):
        raise ValueError("multi-basin static values must have shape [B]")
    return np.asarray(array, dtype=np.float64)


def _write_manifest(
    root: Path,
    *,
    name: str,
    crs: str,
    cadence: str,
    producer_version: str | None,
) -> None:
    if producer_version is None:
        import hydrologeez

        producer_version = f"hydrologeez {hydrologeez.__version__}"
    manifest = {
        "format_version": "0.2",
        "name": name,
        "created_at": datetime.now(UTC).isoformat(),
        "producer_version": producer_version,
        "crs": crs,
        "cadence": cadence,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _derive_cadence(times_by_basin: tuple[np.ndarray, ...]) -> str:
    for basin_times in times_by_basin:
        times = np.sort(basin_times)
        if len(times) < 2:
            continue
        deltas = np.diff(times)
        if np.all(deltas == np.timedelta64(1, "D")):
            continue
        if len(deltas) > 0 and np.all(deltas == deltas[0]):
            raise ValueError("Only daily cadence is supported")
    return "daily"
