"""Bounded HBV output writing and labeled, on-disk selections.

One writer per destination. Completed batches are immutable; incomplete batches
are recomputed. Gregorian ordinal dates follow the CAMELS reader convention.
"""

import fcntl
import hashlib
import json
import os
import shutil
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np
from numpy.typing import NDArray

from .model import Array, Forcing, Memory, Parameters, Result, _validate, simulate


@dataclass(frozen=True)
class Variable:
    units: str
    dimensions: tuple[str, str]
    time: str
    source: str
    column: int | None


def _variables() -> dict[str, Variable]:
    result: dict[str, Variable] = {}
    groups = [
        ("zone", "zone", "end of day", ["snow", "snow_liquid", "soil_moisture"]),
        ("groundwater", "simulation", "end of day", ["upper_storage", "lower_storage"]),
        ("routing", "lag", "end of day", ["routing"]),
        (
            "zone_flux",
            "zone",
            "daily total",
            [
                "soil_input",
                "recharge",
                "evaporation",
                "corrected_precipitation",
                "melt",
                "refreeze",
                "raw_recharge",
                "overflow",
            ],
        ),
        (
            "basin_flux",
            "simulation",
            "daily total",
            ["discharge", "generated_runoff", "percolation", "upper_outflow", "lower_outflow"],
        ),
    ]
    for source, axis, time, names in groups:
        for column, name in enumerate(names):
            result[name] = Variable(
                "mm" if time == "end of day" else "mm/day",
                ("date", axis),
                time,
                source,
                None if source == "routing" else column,
            )
    return result


VARIABLES = _variables()


@dataclass(frozen=True)
class Selection:
    """Packed columns retain requested basin-major, parameter-minor order.

    zone_offsets and routing_offsets delimit each simulation's columns. Local
    zones follow Forcing order; lag 1 is the next outlet arrival day.
    """

    dates: NDArray[np.int64]
    basin_ids: tuple[str, ...]
    parameter_ids: tuple[str, ...]
    zone_offsets: NDArray[np.int64]
    routing_offsets: NDArray[np.int64]
    values: dict[str, Array]
    metadata: dict[str, Variable]
    parameters: Parameters


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _json(path: Path, content: dict[str, Any]) -> None:
    with path.open("w") as stream:
        json.dump(content, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _save(path: Path, value: Array) -> None:
    with path.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _append(stream: BinaryIO, value: Array) -> None:
    """Write one contiguous variable tile; release its buffer before the next."""
    stream.write(np.ascontiguousarray(value).data)


def _fingerprint(arrays: list[NDArray[Any]]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str((array.shape, array.dtype.str)).encode())
        # Buffered iteration also handles basin-major/noncontiguous CAMELS views.
        for chunk in np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"], order="C", buffersize=65536):
            digest.update(cast(NDArray[Any], chunk).tobytes())
    return digest.hexdigest()


def _names(names: list[str], available: tuple[str, ...]) -> None:
    if not names or len(set(names)) != len(names) or not set(names) <= set(available):
        raise ValueError("Select unique, nonempty, available identities or variables")


def _offsets(counts: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.concatenate(([0], np.cumsum(counts)))


def _extract(result: Result, name: str) -> Array:
    if name == "discharge":
        return result.outlet
    assert result.history is not None
    v = VARIABLES[name]
    values = getattr(result.history, v.source)
    return values if v.column is None else values[:, :, v.column]


def run_sweep(
    directory: Path,
    forcing: Forcing,
    parameters: Parameters,
    initial: Memory,
    *,
    basin_ids: list[str],
    parameter_ids: list[str],
    start: str,
    variables: list[str],
    record_start: str | None = None,
    batch_size: int = 256,
    window_size: int = 365,
) -> "OutputStore":
    """Run or resume identical inputs. Existing incompatible destinations fail.

    parameter_ids labels each Parameters row; (basin, parameter) pairs must be
    unique. start labels forcing day zero. No overwrite operation is provided.
    Caller arrays must remain unchanged during execution. A destination admits
    one writer, enforced with a process-lifetime advisory lock (POSIX).
    """
    _names(variables, tuple(VARIABLES))
    _names(basin_ids, tuple(basin_ids))
    first = date.fromisoformat(start).toordinal()
    recorded = first if record_start is None else date.fromisoformat(record_start).toordinal()
    days = len(forcing.values)
    if batch_size <= 0 or window_size <= 0 or not first <= recorded < first + days:
        raise ValueError("Positive batch/window sizes and a recording date within forcing are required")
    if len(basin_ids) != len(forcing.zone_offsets) - 1 or len(parameter_ids) != len(parameters.basin_index):
        raise ValueError("Identity counts must match forcing basins and parameter rows")
    # Validate parameters and memory without allocating histories or scanning all forcing.
    _validate(forcing, parameters, initial, 0, 0)
    pairs = [(basin_ids[int(b)], p) for b, p in zip(parameters.basin_index, parameter_ids, strict=True)]
    if any(not p for p in parameter_ids) or len(set(pairs)) != len(pairs):
        raise ValueError("Nonempty parameter IDs and unique basin/parameter pairs are required")
    zone_offsets = _offsets(np.diff(forcing.zone_offsets)[parameters.basin_index])
    routing_offsets = _offsets(np.ceil(parameters.maxbas).astype(np.int64) - 1)
    manifest = {
        "format": 1,
        "start": first,
        "stop": first + days,
        "record_start": recorded,
        "variables": variables,
        "basin_ids": basin_ids,
        "parameter_ids": parameter_ids,
        "basin_index": parameters.basin_index.tolist(),
        "zone_offsets": zone_offsets.tolist(),
        "routing_offsets": routing_offsets.tolist(),
        "batch_size": batch_size,
        "window_size": window_size,
        "fingerprint": _fingerprint(
            [
                forcing.values,
                forcing.area,
                forcing.zone_offsets,
                parameters.basin_index,
                parameters.zone,
                parameters.groundwater,
                parameters.maxbas,
                initial.zone,
                initial.groundwater,
                initial.routing,
            ]
        ),
        "metadata": json.loads(json.dumps({name: asdict(VARIABLES[name]) for name in variables})),
    }
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".writer"
    lock_stream = lock.open("a")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_stream.close()
        raise ValueError("Output store already has a writer") from None
    try:
        metadata = directory / "metadata.json"
        if metadata.exists():
            if json.loads(metadata.read_text()) != manifest:
                raise ValueError("Incompatible resume inputs or recording settings")
        else:
            if any(p.name != ".writer" for p in directory.iterdir()):
                raise ValueError("Existing destination is not an output store")
            _json(directory / "metadata.tmp", manifest)
            (directory / "metadata.tmp").replace(metadata)
            _sync_directory(directory)
        for a in range(0, len(parameter_ids), batch_size):
            b = min(a + batch_size, len(parameter_ids))
            done = directory / f"batch-{a}"
            if done.exists():
                _check_batch(done)
                continue
            pending = directory / f"pending-{a}"
            if pending.exists():
                shutil.rmtree(pending)
            pending.mkdir()
            zs, ze = zone_offsets[[a, b]]
            rs, re = routing_offsets[[a, b]]
            params = Parameters(
                parameters.basin_index[a:b], parameters.zone[zs:ze], parameters.groundwater[a:b], parameters.maxbas[a:b]
            )
            memory = Memory(initial.zone[zs:ze], initial.groundwater[a:b], initial.routing[rs:re])
            for name, value in [
                ("parameters_zone", params.zone),
                ("parameters_groundwater", params.groundwater),
                ("maxbas", params.maxbas),
            ]:
                _save(pending / f"{name}.npy", value)
            with ExitStack() as stack:
                outputs: dict[str, BinaryIO] = {}
                widths = {"simulation": b - a, "zone": int(ze - zs), "lag": int(re - rs)}
                for name in variables:
                    stream = stack.enter_context((pending / f"{name}.npy").open("wb"))
                    np.lib.format.write_array_header_1_0(
                        stream,
                        {
                            "descr": np.dtype(np.float64).str,
                            "fortran_order": False,
                            "shape": (first + days - recorded, widths[VARIABLES[name].dimensions[1]]),
                        },
                    )
                    outputs[name] = stream
                t = 0
                while t < days:
                    stop = min(t + window_size, days)
                    if t < recorded - first:
                        stop = min(stop, recorded - first)
                    keep = t >= recorded - first
                    result = simulate(
                        forcing, params, memory, start=t, stop=stop, history=keep and variables != ["discharge"]
                    )
                    memory = result.final
                    if keep:
                        for name in variables:
                            _append(outputs[name], _extract(result, name))
                    del result
                    t = stop
                for output in outputs.values():
                    output.flush()
                    os.fsync(output.fileno())
            files = {str(p.relative_to(pending)): p.stat().st_size for p in pending.rglob("*.npy")}
            _json(pending / "complete.json", {"files": files})
            _sync_directory(pending)
            pending.rename(done)
            _sync_directory(directory)
    finally:
        lock_stream.close()
    return OutputStore(directory)


def _check_batch(directory: Path) -> dict[str, Any]:
    record: dict[str, Any] = json.loads((directory / "complete.json").read_text())
    if any(
        not (directory / name).is_file() or (directory / name).stat().st_size != size
        for name, size in record["files"].items()
    ):
        raise ValueError(f"Completed batch has missing or truncated output: {directory}")
    return record


class OutputStore:
    """Metadata-only inspection. Reads require the entire run to be complete."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.metadata: dict[str, Any] = json.loads((directory / "metadata.json").read_text())
        m = self.metadata
        if m["format"] != 1:
            raise ValueError("Unsupported output format")
        self.variables = tuple(m["variables"])
        self.basin_ids = tuple(m["basin_ids"])
        self.parameter_ids = tuple(dict.fromkeys(m["parameter_ids"]))
        self.start = date.fromordinal(m["record_start"]).isoformat()
        self.stop = date.fromordinal(m["stop"]).isoformat()
        self.completed_batches = tuple(
            a for a in range(0, len(m["parameter_ids"]), m["batch_size"]) if (directory / f"batch-{a}").is_dir()
        )
        for a in self.completed_batches:
            _check_batch(directory / f"batch-{a}")
        self.complete = len(self.completed_batches) == len(range(0, len(m["parameter_ids"]), m["batch_size"]))

    def read(
        self, *, variables: list[str], basin_ids: list[str], parameter_ids: list[str], start: str, stop: str
    ) -> Selection:
        """Copy only requested values. Dates are start-inclusive, stop-exclusive.

        Missing combinations fail, including basins without a requested parameter.
        Parameters are returned in the same packed ordering as the selected values.
        """
        if not self.complete:
            raise ValueError("Cannot read an incomplete run; resume it first")
        _names(variables, self.variables)
        _names(basin_ids, self.basin_ids)
        _names(parameter_ids, self.parameter_ids)
        m = self.metadata
        first, last = date.fromisoformat(start).toordinal(), date.fromisoformat(stop).toordinal()
        if not m["record_start"] <= first < last <= m["stop"]:
            raise ValueError("Date selection exceeds recorded coverage or is empty")
        lookup = {
            (self.basin_ids[b], p): i for i, (b, p) in enumerate(zip(m["basin_index"], m["parameter_ids"], strict=True))
        }
        pairs = [(b, p) for b in basin_ids for p in parameter_ids]
        if any(pair not in lookup for pair in pairs):
            raise ValueError("Unavailable basin/parameter combination")
        rows = [lookup[pair] for pair in pairs]
        zo, ro = np.array(m["zone_offsets"]), np.array(m["routing_offsets"])
        zones = _offsets(np.diff(zo)[rows])
        routing = _offsets(np.diff(ro)[rows])
        widths = {"simulation": len(rows), "zone": int(zones[-1]), "lag": int(routing[-1])}
        values = {name: np.empty((last - first, widths[VARIABLES[name].dimensions[1]])) for name in variables}
        pzone, pground, pmax = np.empty((zones[-1], 8)), np.empty((len(rows), 4)), np.empty(len(rows))
        for j, row in enumerate(rows):
            a = row // m["batch_size"] * m["batch_size"]
            batch = self.directory / f"batch-{a}"
            _check_batch(batch)
            zslice = slice(zo[row] - zo[a], zo[row + 1] - zo[a])
            pzone[zones[j] : zones[j + 1]] = np.load(batch / "parameters_zone.npy", mmap_mode="r")[zslice]
            pground[j] = np.load(batch / "parameters_groundwater.npy", mmap_mode="r")[row - a]
            pmax[j] = np.load(batch / "maxbas.npy", mmap_mode="r")[row - a]
            for name in variables:
                axis = VARIABLES[name].dimensions[1]
                if axis == "simulation":
                    src, dst = slice(row - a, row - a + 1), slice(j, j + 1)
                elif axis == "zone":
                    src, dst = zslice, slice(zones[j], zones[j + 1])
                else:
                    src, dst = slice(ro[row] - ro[a], ro[row + 1] - ro[a]), slice(routing[j], routing[j + 1])
                mapped = np.load(batch / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                values[name][:, dst] = mapped[first - m["record_start"] : last - m["record_start"], src]
                del mapped
        return Selection(
            np.arange(first, last, dtype=np.int64),
            tuple(b for b, _ in pairs),
            tuple(p for _, p in pairs),
            zones,
            routing,
            values,
            {name: VARIABLES[name] for name in variables},
            Parameters(np.array([basin_ids.index(b) for b, _ in pairs], dtype=np.int64), pzone, pground, pmax),
        )
