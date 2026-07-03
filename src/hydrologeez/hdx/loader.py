from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from hydrologeez.hdx._polars import require_polars
from hydrologeez.hdx.vocabulary import Vocabulary
from hydrologeez.models.gr6j import GR6JForcing
from hydrologeez.models.hbv import HBVForcing

if TYPE_CHECKING:
    import polars as pl

_REQUIRED_FORCING_FIELDS = {
    GR6JForcing: ("precip", "pet"),
    HBVForcing: ("precip", "pet", "temp"),
}


@dataclass(frozen=True)
class HDXData:
    """Array-native bundle loaded from an HDX scalar dataset.

    Shapes: SINGLE basin (dataset has exactly 1 basin) -> leaves are [T];
    MULTI basin (N>1) -> leaves are [B, T] padded to the max length with 0.0,
    with `mask` [B, T] (True = real record). `times` is per-basin (ragged, NOT
    padded): a 1-D numpy datetime64[us] array [T] for single basin, or a tuple
    of such arrays for multi-basin. `statics` maps each static field name to a
    [B] float64 array (basin-aligned) for multi, or a scalar [] array for single.
    """

    forcing: GR6JForcing | HBVForcing | None
    streamflow: jax.Array | None
    statics: dict[str, jax.Array]
    basin_ids: tuple[str, ...]
    times: np.ndarray | tuple[np.ndarray, ...]
    mask: jax.Array | None


def from_hdx(
    root,
    *,
    forcing_type: type[GR6JForcing] | type[HBVForcing] | None = None,
    vocabulary: Vocabulary | None = None,
    validate: bool = False,
) -> HDXData:
    pl = require_polars()
    vocabulary = Vocabulary() if vocabulary is None else vocabulary
    root_path = Path(root)

    if validate:
        _validate_with_hdx_binary(root_path)

    manifest = json.loads((root_path / "manifest.json").read_text())
    format_version = manifest.get("format_version")
    if format_version != "0.2":
        raise ValueError(f"Unsupported HDX format_version: {format_version!r}")

    static_df = pl.read_parquet(root_path / "scalar_static.parquet")
    basin_ids = tuple(str(basin_id) for basin_id in static_df["basin_id"].to_list())
    statics = _load_statics(static_df, len(basin_ids))

    per_basin_fields: list[dict[str, np.ndarray]] = []
    times_by_basin: list[np.ndarray] = []
    for basin_id in basin_ids:
        dynamic_df = pl.read_parquet(root_path / f"basin={basin_id}" / "scalar_dynamic.parquet").sort("time")
        times_by_basin.append(np.asarray(dynamic_df["time"].to_numpy(), dtype="datetime64[us]"))
        per_basin_fields.append(_load_dynamic_fields(dynamic_df, vocabulary))

    streamflow = _build_field_array(per_basin_fields, "streamflow")
    mask = None if len(basin_ids) == 1 else _build_mask([len(times) for times in times_by_basin])
    forcing = _build_forcing(forcing_type, per_basin_fields)
    times: np.ndarray | tuple[np.ndarray, ...]
    times = times_by_basin[0] if len(times_by_basin) == 1 else tuple(times_by_basin)

    return HDXData(
        forcing=forcing,
        streamflow=streamflow,
        statics=statics,
        basin_ids=basin_ids,
        times=times,
        mask=mask,
    )


def _validate_with_hdx_binary(root: Path) -> None:
    if shutil.which("hdx") is None:
        return
    subprocess.run(["hdx", "validate", str(root)], check=True)


def _load_statics(static_df: pl.DataFrame, basin_count: int) -> dict[str, jax.Array]:
    statics: dict[str, jax.Array] = {}
    for column in static_df.columns:
        if column == "basin_id":
            continue
        values = np.asarray(static_df[column].to_numpy(), dtype=np.float64)
        if basin_count == 1:
            statics[column] = jnp.asarray(values[0], dtype=jnp.float64)
        else:
            statics[column] = jnp.asarray(values, dtype=jnp.float64)
    return statics


def _load_dynamic_fields(dynamic_df: pl.DataFrame, vocabulary: Vocabulary) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    for column in dynamic_df.columns:
        if column in {"basin_id", "time"}:
            continue
        canonical = vocabulary.resolve(column)
        if canonical is None:
            continue
        if vocabulary.role_of(canonical) in {"forcing", "target"}:
            fields[canonical] = np.asarray(dynamic_df[column].to_numpy(), dtype=np.float64)
    return fields


def _build_field_array(per_basin_fields: list[dict[str, np.ndarray]], field_name: str) -> jax.Array | None:
    if not all(field_name in fields for fields in per_basin_fields):
        return None
    values = [fields[field_name] for fields in per_basin_fields]
    if len(values) == 1:
        return jnp.asarray(values[0], dtype=jnp.float64)
    return jnp.asarray(_pad(values), dtype=jnp.float64)


def _build_mask(lengths: list[int]) -> jax.Array:
    max_length = max(lengths, default=0)
    mask = np.zeros((len(lengths), max_length), dtype=np.bool_)
    for basin_index, length in enumerate(lengths):
        mask[basin_index, :length] = True
    return jnp.asarray(mask)


def _build_forcing(
    forcing_type: type[GR6JForcing] | type[HBVForcing] | None,
    per_basin_fields: list[dict[str, np.ndarray]],
) -> GR6JForcing | HBVForcing | None:
    if forcing_type is None:
        return None
    if forcing_type is GR6JForcing:
        if not _has_required_fields(per_basin_fields, _REQUIRED_FORCING_FIELDS[GR6JForcing]):
            return None
        precip = _build_field_array(per_basin_fields, "precip")
        pet = _build_field_array(per_basin_fields, "pet")
        assert precip is not None
        assert pet is not None
        return GR6JForcing(precip=precip, pet=pet)
    if forcing_type is HBVForcing:
        if not _has_required_fields(per_basin_fields, _REQUIRED_FORCING_FIELDS[HBVForcing]):
            return None
        precip = _build_field_array(per_basin_fields, "precip")
        pet = _build_field_array(per_basin_fields, "pet")
        temp = _build_field_array(per_basin_fields, "temp")
        assert precip is not None
        assert pet is not None
        assert temp is not None
        return HBVForcing(precip=precip, pet=pet, temp=temp)
    raise ValueError(f"Unsupported forcing_type: {forcing_type!r}")


def _has_required_fields(per_basin_fields: list[dict[str, np.ndarray]], required_fields: tuple[str, ...]) -> bool:
    return all(all(field_name in fields for field_name in required_fields) for fields in per_basin_fields)


def _pad(values: list[np.ndarray]) -> np.ndarray:
    max_length = max(len(value) for value in values)
    padded = np.zeros((len(values), max_length), dtype=np.float64)
    for basin_index, value in enumerate(values):
        padded[basin_index, : len(value)] = value
    return padded
