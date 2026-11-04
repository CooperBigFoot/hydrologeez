"""Bounded CAMELS/HBV input-path study, not a production input API."""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import resource
import time
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import zarr
from numpy.typing import NDArray
from zarr.codecs import BloscCodec

from hydrologeez.hbv_light import Forcing, Memory, Parameters, Result, simulate
from hydrologeez.hbv_light.model import _validate

Array = NDArray[np.float64]


def oudin(temperature: Array, day: NDArray[np.int64], latitude: float) -> Array:
    from scripts.daymet_prepare import oudin as formula

    return formula(temperature, np.asarray(latitude), day)


def prepare(rows: Array, latitude: float) -> tuple[NDArray[np.int64], Array]:
    dates = np.array([date(int(y), int(m), int(d)).toordinal() for y, m, d in rows[:, :3]], dtype=np.int64)
    if not np.all(np.diff(dates) == 1):
        raise ValueError("Expected contiguous daily Gregorian dates")
    raw = rows[:, [5, 8, 9]]
    if not np.isfinite(raw).all() or np.any(raw[:, 0] < 0) or np.any(raw[:, 1:] <= -999):
        raise ValueError("Invalid/missing precipitation or temperature")
    temperature = (rows[:, 8] + rows[:, 9]) / 2
    doy = np.array([date.fromordinal(int(d)).timetuple().tm_yday for d in dates], dtype=np.int64)
    return dates, np.column_stack((rows[:, 5], temperature, oudin(temperature, doy, latitude)))


def construct(directory: Path, ids: list[str], dates: NDArray[np.int64], values: Array) -> dict[str, float]:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metadata.json").write_text(
        json.dumps({"ids": ids, "start_ordinal": int(dates[0]), "days": len(dates), "units": ["mm/day", "C", "mm/day"]})
    )
    times: dict[str, float] = {}
    for order in ("time", "basin"):
        start = time.perf_counter()
        data = values if order == "time" else np.ascontiguousarray(values.transpose(1, 0, 2))
        times[f"layout-{order}"] = time.perf_counter() - start
        start = time.perf_counter()
        np.save(directory / f"{order}.npy", data)
        times[f"numpy-{order}"] = time.perf_counter() - start
        start = time.perf_counter()
        chunks = (365, 16, 3) if order == "time" else (16, 365, 3)
        target = zarr.create_array(
            str(directory / f"{order}.zarr"),
            overwrite=True,
            shape=data.shape,
            dtype="f8",
            chunks=chunks,
            compressors=[BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")],
        )
        target[:] = data
        times[f"zarr-{order}"] = time.perf_counter() - start
    start = time.perf_counter()
    tables = [
        pa.table(
            {
                "basin": [basin] * len(dates),
                "date": dates,
                **{name: values[:, i, j] for j, name in enumerate(("P", "T", "PET"))},
            }
        )
        for i, basin in enumerate(ids)
    ]
    times["parquet-table-preparation"] = time.perf_counter() - start
    start = time.perf_counter()
    (directory / "basins").mkdir(exist_ok=True)
    for basin, table in zip(ids, tables, strict=True):
        pq.write_table(
            table,
            directory / "basins" / f"{basin}.parquet",
            compression="zstd",
            compression_level=3,
            row_group_size=365,
        )
    times["parquet-files"] = time.perf_counter() - start
    start = time.perf_counter()
    with pq.ParquetWriter(
        directory / "consolidated.parquet", tables[0].schema, compression="zstd", compression_level=3
    ) as writer:
        for table in tables:
            writer.write_table(table, row_group_size=365)
    times["parquet-consolidated"] = time.perf_counter() - start
    return times


def load(
    directory: Path, kind: str, selected: list[str], start: int, stop: int, order: str
) -> tuple[Array, dict[str, float]]:
    begin = time.perf_counter()
    meta = json.loads((directory / "metadata.json").read_text())
    indices = [meta["ids"].index(basin) for basin in selected]
    selection = time.perf_counter() - begin
    begin = time.perf_counter()
    if kind.startswith("parquet"):
        source = (
            [str(directory / "basins" / f"{basin}.parquet") for basin in selected]
            if kind == "parquet-files"
            else str(directory / "consolidated.parquet")
        )
        table = ds.dataset(source, format="parquet").to_table(
            columns=["basin", "date", "P", "T", "PET"],
            filter=ds.field("basin").isin(selected)
            & (ds.field("date") >= (meta["start_ordinal"] + start))
            & (ds.field("date") <= (meta["start_ordinal"] + stop - 1)),
            use_threads=False,
        )
        decoded = time.perf_counter() - begin
        begin = time.perf_counter()
        table = table.sort_by([("basin", "ascending"), ("date", "ascending")])
        expected_dates = np.arange(meta["start_ordinal"] + start, meta["start_ordinal"] + stop)
        sorted_ids = sorted(selected)
        np.testing.assert_array_equal(table["basin"].to_numpy(), np.repeat(sorted_ids, stop - start))
        np.testing.assert_array_equal(table["date"].to_numpy(), np.tile(expected_dates, len(selected)))
        basin_values = np.column_stack([table[name].to_numpy() for name in ("P", "T", "PET")]).reshape(
            len(selected), stop - start, 3
        )
        values = basin_values[[sorted_ids.index(basin) for basin in selected]].transpose(1, 0, 2)
    else:
        physical = kind.split("-")[1]
        if kind.startswith("numpy"):
            source_array = np.load(directory / f"{physical}.npy", mmap_mode="r")
            values = (
                source_array[start:stop][:, indices, :]
                if physical == "time"
                else source_array[indices, start:stop, :].transpose(1, 0, 2)
            )
        else:
            source_array = zarr.open_array(str(directory / f"{physical}.zarr"), mode="r")
            values = (
                np.asarray(source_array.oindex[start:stop, np.asarray(indices), :], dtype=np.float64)
                if physical == "time"
                else np.asarray(source_array.oindex[np.asarray(indices), start:stop, :], dtype=np.float64).transpose(
                    1, 0, 2
                )
            )
        # Force all selected pages/chunks before read timer ends (not mmap-open timing).
        float(np.sum(values))
        decoded = time.perf_counter() - begin
        begin = time.perf_counter()
    packed = (
        np.ascontiguousarray(values)
        if order == "time"
        else np.ascontiguousarray(values.transpose(1, 0, 2)).transpose(1, 0, 2)
    )
    return packed, {"selection": selection, "read_decode_touch": decoded, "pack": time.perf_counter() - begin}


def inputs(values: Array, copies: int) -> tuple[Forcing, Parameters, Memory]:
    basins = values.shape[1]
    count = basins * copies
    forcing = Forcing(values, np.ones(basins), np.arange(basins + 1, dtype=np.int64))
    parameters = Parameters(
        np.tile(np.arange(basins, dtype=np.int64), copies),
        np.tile([0.0, 2.0, 0.05, 0.1, 1.0, 100.0, 1.0, 0.5], (count, 1)),
        np.tile([1.0, 0.5, 0.1, 0.05], (count, 1)),
        np.full(count, 2.5),
    )
    initial = Memory(np.tile([0.0, 0.0, 50.0], (count, 1)), np.zeros((count, 2)), np.zeros(count * 2))
    return forcing, parameters, initial


def consume(values: Array, copies: int) -> tuple[Result, dict[str, float]]:
    forcing, parameters, initial = inputs(values, copies)
    begin = time.perf_counter()
    _validate(forcing, parameters, initial, 0, len(values))
    validation = time.perf_counter() - begin
    begin = time.perf_counter()
    result = simulate(forcing, parameters, initial, history=False)
    elapsed = time.perf_counter() - begin
    return result, {"validation_probe": validation, "simulate_including_validation": elapsed}


def equivalent(actual: Result, expected: Result) -> None:
    np.testing.assert_array_equal(actual.outlet, expected.outlet)
    for name in ("zone", "groundwater", "routing"):
        np.testing.assert_array_equal(getattr(actual.final, name), getattr(expected.final, name))


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    numba.set_num_threads(1)
    pa.set_cpu_count(1)
    zarr.config.set({"async.concurrency": 1, "threading.max_workers": 1, "codec_pipeline.max_workers": 1})
    begin = time.perf_counter()
    attributes = list(csv.DictReader((args.raw / "camels_topo.txt").read_text().splitlines(), delimiter=";"))
    attributes.sort(key=lambda row: row["gauge_id"])
    chosen = [attributes[i] for i in np.linspace(0, len(attributes) - 1, 128, dtype=int)]
    ids = [row["gauge_id"] for row in chosen]
    prepared = []
    reference_dates = np.arange(date(1990, 1, 1).toordinal(), date(2000, 1, 1).toordinal(), dtype=np.int64)
    with zipfile.ZipFile(args.raw / "basin_timeseries_v1p2_metForcing_obsFlow.zip") as archive:
        members = {
            Path(name).name[:8]: name
            for name in archive.namelist()
            if "/basin_mean_forcing/daymet/" in name and name.endswith("_lump_cida_forcing_leap.txt")
        }
        for row in chosen:
            text = archive.read(members[row["gauge_id"]]).decode()
            if text.splitlines()[3].split() != [
                "Year",
                "Mnth",
                "Day",
                "Hr",
                "dayl(s)",
                "prcp(mm/day)",
                "srad(W/m2)",
                "swe(mm)",
                "tmax(C)",
                "tmin(C)",
                "vp(Pa)",
            ]:
                raise ValueError("Unexpected forcing columns/units")
            dates, values = prepare(np.loadtxt(io.StringIO(text), skiprows=4), float(row["gauge_lat"]))
            mask = (dates >= reference_dates[0]) & (dates <= reference_dates[-1])
            np.testing.assert_array_equal(dates[mask], reference_dates)
            prepared.append(values[mask])
    canonical = np.stack(prepared, axis=1)
    preparation = time.perf_counter() - begin
    construction = construct(args.output, ids, reference_dates, canonical)
    records: list[dict[str, Any]] = []
    # Compile C and physically basin-major signatures separately; never count JIT as runtime.
    begin = time.perf_counter()
    consume(canonical[:10, :2].copy(), 1)
    consume(np.ascontiguousarray(canonical[:10, :2].transpose(1, 0, 2)).transpose(1, 0, 2), 1)
    compilation = time.perf_counter() - begin
    cases = [
        ("few-short", list(range(4)), 0, 365),
        ("many-short", list(range(128)), 0, 365),
        ("contiguous-long", list(range(32)), 0, len(canonical)),
        ("scattered-long", list(range(0, 128, 4)), 0, len(canonical)),
        ("many-long", list(range(128)), 0, len(canonical)),
    ]
    kinds = ["numpy-time", "numpy-basin", "zarr-time", "zarr-basin", "parquet-files", "parquet-consolidated"]
    for label, indices, start, stop in cases:
        expected = canonical[start:stop][:, indices]
        reference, _ = consume(expected.copy(), 4)
        for order in ("time", "basin"):
            resident = (
                np.ascontiguousarray(expected)
                if order == "time"
                else np.ascontiguousarray(expected.transpose(1, 0, 2)).transpose(1, 0, 2)
            )
            for repeat in range(4):
                outlet, times = consume(resident, 4)
                equivalent(outlet, reference)
                records.append(
                    {
                        "case": label,
                        "kind": "resident",
                        "order": order,
                        "repeat": repeat,
                        "strides": resident.strides,
                        **times,
                    }
                )
            for kind in kinds:
                for repeat in range(4):
                    begin = time.perf_counter()
                    loaded, stages = load(args.output, kind, [ids[i] for i in indices], start, stop, order)
                    loading = time.perf_counter() - begin
                    np.testing.assert_array_equal(loaded, expected)
                    outlet, times = consume(loaded, 4)
                    equivalent(outlet, reference)
                    records.append(
                        {
                            "case": label,
                            "kind": kind,
                            "order": order,
                            "repeat": repeat,
                            "strides": loaded.strides,
                            "input_bytes": loaded.nbytes,
                            "load_total": loading,
                            "total": loading + times["simulate_including_validation"],
                            **stages,
                            **times,
                        }
                    )
        print(label, flush=True)
    sizes = {str(p.relative_to(args.output)): p.stat().st_size for p in args.output.rglob("*") if p.is_file()}
    _, parameters, _ = inputs(canonical, 4)
    evidence = {
        "platform": platform.platform(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "arrow": pa.__version__,
            "zarr": zarr.__version__,
        },
        "threads": 1,
        "zarr_concurrency": 1,
        "basins": ids,
        "date_window": ["1990-01-01", "2000-01-01 exclusive"],
        "parameter_copies": 4,
        "groundwater_formulation": "nonlinear upper, linear lower; recharge then percolation then discharge",
        "zone_parameters_TT_CFMAX_CFR_CWH_SCF_FC_BETA_LP": parameters.zone[0].tolist(),
        "groundwater_parameters_PERC_Alpha_K1_K2": parameters.groundwater[0].tolist(),
        "maxbas_days": float(parameters.maxbas[0]),
        "preparation_seconds": preparation,
        "construction_seconds": construction,
        "compilation_seconds": compilation,
        "peak_rss_native": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "stored_files": sizes,
        "records": records,
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(evidence, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
