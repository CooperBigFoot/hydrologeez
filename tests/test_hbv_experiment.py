from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from experiments.hbv.model import day, weights


def test_worked_day() -> None:
    zone = np.array([[10.0, 0.0, 50.0]])
    basin = np.array([10.0, 20.0, 0.0])
    parameters = np.array([[0.0, 2.0, 0.05, 0.1, 1.0, 100.0, 1.0, 0.5]])
    response = np.array([1.0, 12.0, 0.2, 0.1, 0.05])
    flux, runoff = day(np.array([[6.0, 2.0, 2.0]]), np.ones(1), parameters, response, weights(2.0), zone, basin)
    assert_allclose(zone, [[6.0, 0.6, 52.7]])
    assert_allclose(basin, [11.99, 19.95, 1.38])
    assert_allclose(flux[0, :3], [9.4, 4.7, 2.0])
    assert_allclose(runoff, [1.38, 2.76, 1.0, 0.34, 1.37, 1.05])


def test_fractional_routing_weights() -> None:
    assert_allclose(weights(2.5), [0.32, 0.60, 0.08])
    assert_allclose(weights(1.0), [1.0])


def test_boundaries_and_area_weighting() -> None:
    parameters = np.array([[0.0, 2.0, 0.5, 0.0, 1.0, 100.0, 1.0, 0.5]] * 2)
    zone = np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 100.0]])
    basin = np.zeros(2)
    flux, runoff = day(
        np.array([[8.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.array([0.25, 0.75]),
        parameters,
        np.array([0.0, 0.0, 1.0, 1.0, 0.0]),
        weights(1.0),
        zone,
        basin,
    )
    assert_allclose(flux[:, 1], [8.0, 2.0])
    assert_allclose(runoff, [3.5, 3.5, 0.0, 1.75, 1.75, 0.0])
    assert_allclose(basin, 0.0)
    zone[:] = 0.0
    flux, _ = day(
        np.array([[150.0, 0.0, 0.0], [0.0, 0.0, 100.0]]),
        np.array([0.5, 0.5]),
        parameters,
        np.zeros(5),
        weights(1.0),
        zone,
        basin,
    )
    assert_allclose(zone[:, 2], [100.0, 0.0])
    assert_allclose(flux[:, 1], [50.0, 0.0])


def test_restart_history_and_parallel() -> None:
    from experiments.hbv.execution import simulate
    from experiments.hbv.inputs import synthetic

    data = synthetic(4, 3, 40, seed=8)
    whole = simulate(data, history=True)
    first = simulate(data, stop=17, history=True)
    second = simulate(data, start=17, initial=(first.zone, first.basin), history=True)
    assert_allclose(np.concatenate([first.prediction, second.prediction], axis=1), whole.prediction)
    assert_allclose(second.zone, whole.zone)
    assert_allclose(second.basin, whole.basin)
    assert_allclose(first.zone_history[:, 0], data.zone)
    assert_allclose(whole.zone_history[:, -1], whole.zone)
    parallel = simulate(data, parallel=True, history=True)
    assert_allclose(parallel.prediction, whole.prediction)
    assert_allclose(parallel.zone_flux, whole.zone_flux)
    assert_allclose(parallel.basin_flux, whole.basin_flux)
    from experiments.hbv.inputs import expanded_forcing

    expanded = simulate(expanded_forcing(data), history=True)
    assert_allclose(expanded.prediction, whole.prediction)
    assert_allclose(expanded.zone_history, whole.zone_history)
    assert_allclose(expanded.basin_flux, whole.basin_flux)


def test_parquet_chunk_boundary(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    from experiments.hbv.execution import simulate
    from experiments.hbv.inputs import synthetic
    from experiments.hbv.recording import record

    data = synthetic(3, 2, 9)
    expected = simulate(data, history=True)
    record(data, tmp_path, chunk_days=4)
    table = pq.read_table(tmp_path / "basin.parquet")
    mask = (table["basin_id"].to_numpy() == 1) & (table["candidate"].to_numpy() == 0)
    rows = table.filter(mask).sort_by("state_day")
    assert_allclose(rows["U"].to_numpy(), expected.basin_history[2, :, 0])
    assert_allclose(rows["h0"].to_numpy(), expected.basin_history[2, :, 2])
    assert_allclose(rows["Q"].to_numpy()[1:], expected.prediction[2])
    assert rows["state_day"].to_pylist() == list(range(10))
    zones = pq.read_table(tmp_path / "zone.parquet")
    mask = (
        (zones["basin_id"].to_numpy() == 1) & (zones["candidate"].to_numpy() == 0) & (zones["zone_id"].to_numpy() == 1)
    )
    rows = zones.filter(mask).sort_by("state_day")
    assert_allclose(rows["M"].to_numpy(), expected.zone_history[2, :, 1, 2])


def test_independent_nonlinear_and_snow_cases() -> None:
    params = np.array([[0.0, 2.0, 0.5, 0.25, 2.0, 100.0, 2.0, 0.5]])
    zone = np.array([[4.0, 1.0, 50.0]])
    basin = np.zeros(2)
    flux, _ = day(np.array([[3.0, -2.0, 0.0]]), np.ones(1), params, np.zeros(5), weights(1.0), zone, basin)
    assert_allclose(zone[0, :2], [11.0, 0.0])
    assert_allclose(flux[0, [0, 3, 5]], [0.0, 6.0, 1.0])
    day(np.array([[0.0, 2.0, 0.0]]), np.ones(1), params, np.zeros(5), weights(1.0), zone, basin)
    assert_allclose(zone[0, :2], [7.0, 1.75])
    zone[:] = [0.0, 0.0, 50.0]
    flux, _ = day(np.array([[80.0, 0.0, 10.0]]), np.ones(1), params, np.zeros(5), weights(1.0), zone, basin)
    assert_allclose(flux[0, [1, 2, 6, 7]], [30.0, 10.0, 20.0, 10.0])
    assert_allclose(zone[0, 2], 90.0)


def test_sharing_fixed_values_and_water_balance() -> None:
    from experiments.hbv.execution import simulate
    from experiments.hbv.inputs import synthetic

    data = synthetic(6, 3, 400)
    assert_allclose(data.parameters[:, 0, 1], data.parameters[:, 2, 1])
    assert_allclose(data.parameters[6], data.parameters[7])  # all fixed basin
    assert_allclose(data.response[6], data.response[8])
    result = simulate(data, history=True)
    assert not np.allclose(result.zone_history[:, :, 0], result.zone_history[:, :, 2])
    area = data.area[data.basin_index]
    water = (result.zone_history.sum(axis=3) * area[:, None, :]).sum(axis=2) + result.basin_history.sum(axis=2)
    external = ((result.zone_flux[..., 3] - result.zone_flux[..., 2]) * area[:, None, :]).sum(axis=2)
    assert_allclose(np.diff(water, axis=1), external - result.prediction, atol=2e-13)
    original = result.prediction.copy()
    data.parameters[6:9, :, 0] += 3.0  # change an explicit fixed input, not a scientific constant
    changed = simulate(data)
    assert not np.allclose(changed.prediction[6:9], original[6:9])
    assert_allclose(changed.prediction[:6], original[:6])


def test_expanded_disk_preserves_basin_identity(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    from experiments.hbv.inputs import expanded_forcing, synthetic
    from experiments.hbv.recording import record

    data = synthetic(2, 3, 2)
    record(expanded_forcing(data), tmp_path, chunk_days=1)
    table = pq.read_table(tmp_path / "basin.parquet")
    initial = table.filter(table["state_day"].to_numpy() == 0)
    assert initial["basin_id"].to_pylist() == [0, 0, 0, 1, 1, 1]
    assert initial["candidate"].to_pylist() == [0, 1, 2, 0, 1, 2]
    with np.load(tmp_path / "inputs.npz") as saved:
        assert_allclose(saved["basin_index"], data.basin_index)
        assert_allclose(saved["fixed"][saved["basin_index"]], data.fixed[data.basin_index])
