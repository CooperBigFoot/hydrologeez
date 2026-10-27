"""Experimental chunked Parquet output, deliberately outside model calculations."""

from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .execution import simulate
from .inputs import Inputs


def record(data: Inputs, directory: Path, chunk_days: int = 16) -> dict[str, float]:
    directory.mkdir(parents=True, exist_ok=True)
    # All inputs and identities needed to interpret or reproduce any saved state.
    np.savez(directory / "inputs.npz", **vars(data))
    writers: dict[str, pq.ParquetWriter] = {}
    timings = {"calculation": 0.0, "materialization": 0.0, "disk": 0.0}
    initial = None
    n, z, _ = data.zone.shape
    try:
        for start in range(0, data.forcing.shape[1], chunk_days):
            stop = min(start + chunk_days, data.forcing.shape[1])
            tick = perf_counter()
            result = simulate(data, start=start, stop=stop, initial=initial, history=True)
            timings["calculation"] += perf_counter() - tick
            initial = result.zone, result.basin
            tick = perf_counter()
            first = 0 if start == 0 else 1
            days = np.arange(start + first, stop + 1)
            width = len(days)
            basin_columns = {
                "basin_id": np.repeat(data.basin_index, width),
                "candidate": np.repeat(data.candidate, width),
                "state_day": np.tile(days, n),
            }
            for j, name in enumerate(["U", "V"] + [f"h{i}" for i in range(data.basin.shape[1] - 2)]):
                basin_columns[name] = result.basin_history[:, first:, j].ravel()
            for j, name in enumerate(["Q", "G", "D", "q0", "q1", "q2"]):
                values = result.basin_flux[:, :, j]
                if start == 0:
                    values = np.concatenate([np.full((n, 1), np.nan), values], axis=1)
                basin_columns[name] = values.ravel()
            zone_columns = {
                "basin_id": np.repeat(data.basin_index, width * z),
                "candidate": np.repeat(data.candidate, width * z),
                "state_day": np.tile(np.repeat(days, z), n),
                "zone_id": np.tile(np.arange(z), n * width),
                "area_fraction": np.repeat(data.area[data.forcing_index, None, :], width, axis=1).ravel(),
            }
            for j, name in enumerate(["S", "L", "M"]):
                zone_columns[name] = result.zone_history[:, first:, :, j].ravel()
            for j, name in enumerate(["I", "R", "A", "P_corrected", "melt", "refreeze", "raw_recharge", "overflow"]):
                values = result.zone_flux[:, :, :, j]
                if start == 0:
                    values = np.concatenate([np.full((n, 1, z), np.nan), values], axis=1)
                zone_columns[name] = values.ravel()
            tables = {"basin": pa.table(basin_columns), "zone": pa.table(zone_columns)}
            timings["materialization"] += perf_counter() - tick
            tick = perf_counter()
            for name, table in tables.items():
                if name not in writers:
                    writers[name] = pq.ParquetWriter(directory / f"{name}.parquet", table.schema, compression="zstd")
                writers[name].write_table(table)
            timings["disk"] += perf_counter() - tick
    finally:
        tick = perf_counter()
        for writer in writers.values():
            writer.close()
        timings["disk"] += perf_counter() - tick
    return timings
