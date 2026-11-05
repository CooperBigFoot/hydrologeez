"""Exact readback against the existing model, including packed axes."""

from pathlib import Path
from typing import BinaryIO, TypedDict

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from hydrologeez.hbv_light import Forcing, Memory, Parameters, simulate


class RunOptions(TypedDict):
    basin_ids: list[str]
    parameter_ids: list[str]
    start: str
    variables: list[str]
    batch_size: int
    window_size: int


class ReadOptions(TypedDict):
    variables: list[str]
    basin_ids: list[str]
    parameter_ids: list[str]
    start: str
    stop: str


def inputs() -> tuple[Forcing, Parameters, Memory]:
    values = np.array(
        [[[8.0, -2.0, 0.0], [3.0, 2.0, 1.0], [5.0, 4.0, 2.0]], [[0.0, 5.0, 1.0], [1.0, -1.0, 0.0], [9.0, 8.0, 2.0]]] * 4
    )
    forcing = Forcing(values, np.array([1.0, 0.3, 0.7]), np.array([0, 1, 3]))
    parameters = Parameters(
        np.arange(8) % 2,
        np.tile([0.0, 2.0, 0.05, 0.1, 1.0, 100.0, 1.0, 0.5], (12, 1)),
        np.tile([1.0, 0.5, 0.1, 0.05], (8, 1)),
        np.array([1.0, 2.5] * 4),
    )
    initial = Memory(np.tile([0.0, 0.0, 50.0], (12, 1)), np.zeros((8, 2)), np.zeros(8))
    return forcing, parameters, initial


def test_selected_outputs(tmp_path: Path) -> None:
    from hydrologeez.hbv_light.output import OutputStore, run_sweep

    forcing, parameters, initial = inputs()
    reference = simulate(forcing, parameters, initial, history=True)
    run_sweep(
        tmp_path / "run",
        forcing,
        parameters,
        initial,
        basin_ids=["a", "b"],
        parameter_ids=[str(i // 2) for i in range(8)],
        start="2000-01-01",
        record_start="2000-01-03",
        variables=["discharge", "soil_moisture", "routing"],
        batch_size=3,
        window_size=3,
    )
    store = OutputStore(tmp_path / "run")
    assert store.complete
    assert len(list((tmp_path / "run" / "batch-0").glob("discharge.npy"))) == 1
    assert store.variables == ("discharge", "soil_moisture", "routing")
    selected = store.read(
        variables=["discharge", "soil_moisture", "routing"],
        basin_ids=["b", "a"],
        parameter_ids=["3", "0"],
        start="2000-01-04",
        stop="2000-01-08",
    )
    assert selected.basin_ids == ("b", "b", "a", "a")
    assert selected.parameter_ids == ("3", "0", "3", "0")
    assert selected.dates.tolist() == list(range(730123, 730127))
    rows = [7, 1, 6, 0]
    assert_array_equal(selected.values["discharge"], reference.outlet[3:7, rows])
    assert reference.history is not None
    for name, offsets, expected in [
        ("soil_moisture", reference.zone_offsets, reference.history.zone[:, :, 2]),
        ("routing", reference.routing_offsets, reference.history.routing),
    ]:
        columns = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in rows])
        assert_array_equal(selected.values[name], expected[3:7, columns])
    assert selected.metadata["soil_moisture"].time == "end of day"
    assert selected.metadata["discharge"].units == "mm/day"
    assert not list((tmp_path / "run").rglob("snow.npy"))


def test_failure_resume_and_incompatible_inputs(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    from hydrologeez.hbv_light import output

    forcing, parameters, initial = inputs()
    kwargs = RunOptions(
        basin_ids=["a", "b"],
        parameter_ids=[str(i // 2) for i in range(8)],
        start="2000-01-01",
        variables=["discharge", "melt"],
        batch_size=3,
        window_size=3,
    )
    original = output._append
    calls = 0

    def fail(stream: BinaryIO, value: np.ndarray) -> None:
        nonlocal calls
        calls += 1
        original(stream, value)
        if calls == 9:
            raise OSError("injected partial write")

    with monkeypatch.context() as patch:
        patch.setattr(output, "_append", fail)
        with pytest.raises(OSError, match="injected"):
            output.run_sweep(tmp_path / "resumed", forcing, parameters, initial, **kwargs)
    incomplete = output.OutputStore(tmp_path / "resumed")
    assert incomplete.completed_batches == (0,)
    assert not incomplete.complete
    with pytest.raises(ValueError, match="incomplete"):
        incomplete.read(
            variables=["discharge"], basin_ids=["a"], parameter_ids=["0"], start="2000-01-01", stop="2000-01-02"
        )
    saved = {
        str(p): (p.stat().st_mtime_ns, p.read_bytes())
        for p in (tmp_path / "resumed" / "batch-0").rglob("*")
        if p.is_file()
    }
    forcing.values[0, 0, 0] += 1
    with pytest.raises(ValueError, match="Incompatible"):
        output.run_sweep(tmp_path / "resumed", forcing, parameters, initial, **kwargs)
    forcing.values[0, 0, 0] -= 1
    output.run_sweep(tmp_path / "resumed", forcing, parameters, initial, **kwargs)
    output.run_sweep(tmp_path / "whole", forcing, parameters, initial, **kwargs)
    for path, (mtime, data) in saved.items():
        assert Path(path).stat().st_mtime_ns == mtime
        assert Path(path).read_bytes() == data
    for path in (tmp_path / "whole").rglob("*.npy"):
        assert_array_equal(np.load(path), np.load(tmp_path / "resumed" / path.relative_to(tmp_path / "whole")))
    assert output.OutputStore(tmp_path / "resumed").complete
    (tmp_path / "resumed" / "batch-0" / "discharge.npy").unlink()
    with pytest.raises(ValueError, match="missing or truncated"):
        output.run_sweep(tmp_path / "resumed", forcing, parameters, initial, **kwargs)


def test_all_variables_and_lazy_reads(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    from hydrologeez.hbv_light import output

    forcing, parameters, initial = inputs()
    reference = simulate(forcing, parameters, initial, history=True)
    output.run_sweep(
        tmp_path / "run",
        forcing,
        parameters,
        initial,
        basin_ids=["a", "b"],
        parameter_ids=[str(i // 2) for i in range(8)],
        start="2000-01-01",
        variables=list(output.VARIABLES),
        batch_size=3,
        window_size=3,
    )
    original = np.load
    loaded: list[Path] = []

    def load(path: Path, *, mmap_mode: str | None = None, allow_pickle: bool = False) -> np.ndarray:
        assert mmap_mode == "r"
        loaded.append(path)
        return original(path, mmap_mode=mmap_mode, allow_pickle=allow_pickle)

    with monkeypatch.context() as patch:
        patch.setattr(np, "load", load)
        store = output.OutputStore(tmp_path / "run")
        assert not loaded
        small = store.read(
            variables=["discharge"], basin_ids=["b"], parameter_ids=["3"], start="2000-01-08", stop="2000-01-09"
        )
        assert_array_equal(small.values["discharge"], reference.outlet[7:8, 7:8])
        assert all("batch-6" in str(path) for path in loaded)
        assert len(loaded) == 4
    selected = store.read(
        variables=list(output.VARIABLES),
        basin_ids=["a", "b"],
        parameter_ids=["0", "1", "2", "3"],
        start="2000-01-01",
        stop="2000-01-09",
    )
    rows = [0, 2, 4, 6, 1, 3, 5, 7]
    assert reference.history is not None
    expected = {
        "snow": reference.history.zone[:, :, 0],
        "snow_liquid": reference.history.zone[:, :, 1],
        "soil_moisture": reference.history.zone[:, :, 2],
        "upper_storage": reference.history.groundwater[:, :, 0],
        "lower_storage": reference.history.groundwater[:, :, 1],
        "routing": reference.history.routing,
    }
    for i, name in enumerate(
        [
            "soil_input",
            "recharge",
            "evaporation",
            "corrected_precipitation",
            "melt",
            "refreeze",
            "raw_recharge",
            "overflow",
        ]
    ):
        expected[name] = reference.history.zone_flux[:, :, i]
    for i, name in enumerate(["discharge", "generated_runoff", "percolation", "upper_outflow", "lower_outflow"]):
        expected[name] = reference.history.basin_flux[:, :, i]
    for name, value in expected.items():
        if name in [
            "snow",
            "snow_liquid",
            "soil_moisture",
            "soil_input",
            "recharge",
            "evaporation",
            "corrected_precipitation",
            "melt",
            "refreeze",
            "raw_recharge",
            "overflow",
        ]:
            offsets = reference.zone_offsets
        elif name == "routing":
            offsets = reference.routing_offsets
        else:
            offsets = np.arange(9)
        columns = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in rows])
        assert_array_equal(selected.values[name], value[:, columns])
    assert_array_equal(selected.zone_offsets, [0, 1, 2, 3, 4, 6, 8, 10, 12])
    assert_array_equal(selected.routing_offsets, [0, 0, 0, 0, 0, 2, 4, 6, 8])
    assert_array_equal(selected.parameters.maxbas, parameters.maxbas[rows])
    for options in [
        ReadOptions(variables=["unknown"], basin_ids=["a"], parameter_ids=["0"], start="2000-01-01", stop="2000-01-02"),
        ReadOptions(
            variables=["discharge"], basin_ids=["missing"], parameter_ids=["0"], start="2000-01-01", stop="2000-01-02"
        ),
        ReadOptions(
            variables=["discharge"], basin_ids=["a"], parameter_ids=["missing"], start="2000-01-01", stop="2000-01-02"
        ),
        ReadOptions(
            variables=["discharge"], basin_ids=["a"], parameter_ids=["0"], start="1999-12-31", stop="2000-01-02"
        ),
        ReadOptions(
            variables=["discharge"], basin_ids=["a"], parameter_ids=["0"], start="2000-01-01", stop="2000-01-10"
        ),
    ]:
        with pytest.raises(ValueError):
            store.read(**options)


@pytest.mark.parametrize(
    "changed", ["initial", "parameters", "ids", "start", "variables", "record_start", "window_size", "batch_size"]
)
def test_resume_settings(tmp_path: Path, changed: str) -> None:
    from hydrologeez.hbv_light.output import run_sweep

    forcing, parameters, initial = inputs()
    options = RunOptions(
        basin_ids=["a", "b"],
        parameter_ids=[str(i // 2) for i in range(8)],
        start="2000-01-01",
        variables=["discharge"],
        batch_size=3,
        window_size=3,
    )
    run_sweep(tmp_path / "run", forcing, parameters, initial, **options)
    record_start = None
    if changed == "initial":
        initial.groundwater[0, 0] = 1.0
    elif changed == "parameters":
        parameters.groundwater[0, 0] = 2.0
    elif changed == "ids":
        options["basin_ids"] = ["b", "a"]
    elif changed == "start":
        options["start"] = "2000-01-02"
    elif changed == "variables":
        options["variables"] = ["snow"]
    elif changed == "record_start":
        record_start = "2000-01-02"
    elif changed == "window_size":
        options["window_size"] = 4
    else:
        options["batch_size"] = 4
    with pytest.raises(ValueError, match="Incompatible"):
        run_sweep(tmp_path / "run", forcing, parameters, initial, record_start=record_start, **options)


def test_commit_failure_and_missing_combination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hydrologeez.hbv_light.output import OutputStore, run_sweep

    forcing, parameters, initial = inputs()
    options = RunOptions(
        basin_ids=["a", "b"],
        parameter_ids=["unique"] + [str(i // 2) for i in range(1, 8)],
        start="2000-01-01",
        variables=["discharge"],
        batch_size=3,
        window_size=3,
    )
    rename = Path.rename

    def fail(path: Path, target: Path) -> Path:
        if path.name == "pending-3":
            raise OSError("commit failed")
        return rename(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", fail)
        with pytest.raises(OSError, match="commit failed"):
            run_sweep(tmp_path / "run", forcing, parameters, initial, **options)
    assert OutputStore(tmp_path / "run").completed_batches == (0,)
    store = run_sweep(tmp_path / "run", forcing, parameters, initial, **options)
    with pytest.raises(ValueError, match="combination"):
        store.read(
            variables=["discharge"],
            basin_ids=["a", "b"],
            parameter_ids=["unique"],
            start="2000-01-01",
            stop="2000-01-02",
        )


def test_writer_lock_and_existing_destination(tmp_path: Path) -> None:
    import fcntl

    from hydrologeez.hbv_light.output import run_sweep

    forcing, parameters, initial = inputs()
    options = RunOptions(
        basin_ids=["a", "b"],
        parameter_ids=[str(i // 2) for i in range(8)],
        start="2000-01-01",
        variables=["discharge"],
        batch_size=3,
        window_size=3,
    )
    destination = tmp_path / "run"
    destination.mkdir()
    with (destination / ".writer").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="writer"):
            run_sweep(destination, forcing, parameters, initial, **options)
    sentinel = destination / "user-data"
    sentinel.write_text("preserve me")
    with pytest.raises(ValueError, match="Existing destination"):
        run_sweep(destination, forcing, parameters, initial, **options)
    assert sentinel.read_text() == "preserve me"
