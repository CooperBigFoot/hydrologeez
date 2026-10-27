import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_worked_day() -> None:
    from hydrologeez.hbv_light import Forcing, Memory, Parameters, simulate

    forcing = Forcing(np.array([[[6.0, 2.0, 2.0]]]), np.ones(1), np.array([0, 1]))
    parameters = Parameters(
        np.array([0]),
        np.array([[0.0, 2.0, 0.05, 0.1, 1.0, 100.0, 1.0, 0.5]]),
        np.array([[1.0, 12.0, 0.2, 0.1, 0.05]]),
        np.array([2.0]),
    )
    initial = Memory(np.array([[10.0, 0.0, 50.0]]), np.array([[10.0, 20.0]]), np.zeros(1))
    result = simulate(forcing, parameters, initial, history=True)
    assert_allclose(result.outlet, [[1.38]])
    assert_allclose(result.final.zone, [[6.0, 0.6, 52.7]])
    assert_allclose(result.final.groundwater, [[11.99, 19.95]])
    assert_allclose(result.final.routing, [1.38])
    assert result.history is not None
    assert_allclose(result.history.zone, [[[6.0, 0.6, 52.7]]])
    assert_allclose(result.history.zone_flux, [[[9.4, 4.7, 2.0, 6.0, 4.0, 0.0, 4.7, 0.0]]])
    assert_allclose(result.history.basin_flux, [[[1.38, 2.76, 1.0, 0.34, 1.37, 1.05]]])
    assert_allclose(initial.zone, [[10.0, 0.0, 50.0]])


def test_fractional_routing_restart_and_associations() -> None:
    from hydrologeez.hbv_light import Forcing, Memory, Parameters, simulate

    values = np.zeros((3, 3, 3))
    values[0, :, 0] = [8.0, 2.0, 10.0]
    forcing = Forcing(values, np.array([0.25, 0.75, 1.0]), np.array([0, 2, 3]))
    params = Parameters(
        np.array([1, 0, 0]),
        np.tile([0.0, 2.0, 0.5, 0.0, 1.0, 100.0, 1.0, 0.5], (5, 1)),
        np.tile([0.0, 0.0, 1.0, 1.0, 0.0], (3, 1)),
        np.array([2.5, 1.0, 6.5]),
    )
    initial = Memory(np.tile([0.0, 0.0, 100.0], (5, 1)), np.zeros((3, 2)), np.zeros(8))
    whole = simulate(forcing, params, initial, history=True)
    assert_allclose(whole.outlet[:, 0], [3.2, 6.0, 0.8])
    assert_allclose(whole.outlet[:, 1], [3.5, 0.0, 0.0])
    assert_allclose(whole.zone_offsets, [0, 1, 3, 5])
    assert_allclose(whole.routing_offsets, [0, 2, 2, 8])
    assert_allclose(whole.basin_index, [1, 0, 0])
    first = simulate(forcing, params, initial, stop=1, history=True)
    tail = simulate(forcing, params, first.final, start=1, history=True)
    assert first.history is not None and tail.history is not None and whole.history is not None
    for name in ("zone", "groundwater", "routing", "zone_flux", "basin_flux"):
        assert_allclose(
            np.concatenate([getattr(first.history, name), getattr(tail.history, name)]), getattr(whole.history, name)
        )
    assert_allclose(np.concatenate([first.outlet, tail.outlet]), whole.outlet)
    for name in ("zone", "groundwater", "routing"):
        assert_allclose(getattr(tail.final, name), getattr(whole.final, name))
    prediction = simulate(forcing, params, initial)
    assert prediction.history is None
    assert_allclose(prediction.outlet, whole.outlet)


def test_invalid_inputs_are_rejected() -> None:
    from hydrologeez.hbv_light import Forcing, Memory, Parameters, simulate

    forcing = Forcing(np.zeros((1, 1, 3)), np.ones(1), np.array([0, 1]))
    params = Parameters(
        np.array([0]), np.array([[0.0, 2.0, 0.5, 0.1, 1.0, 100.0, 1.0, 0.5]]), np.zeros((1, 5)), np.ones(1)
    )
    initial = Memory(np.zeros((1, 3)), np.zeros((1, 2)), np.zeros(0))
    for array, column, value in [
        (forcing.values, 0, -1.0),
        (forcing.values, 1, np.nan),
        (forcing.area, 0, 0.5),
        (params.zone, 5, 0.0),
        (params.zone, 7, 1.1),
        (params.groundwater, 0, -1.0),
        (params.maxbas, 0, 0.9),
        (initial.zone, 1, 1.0),
    ]:
        old = array[..., column].copy()
        array[..., column] = value
        with pytest.raises(ValueError):
            simulate(forcing, params, initial)
        array[..., column] = old
    with pytest.raises(ValueError):
        simulate(forcing, params, initial, stop=2)


@pytest.mark.parametrize(
    "rain, temperature, pet, state, expected_state, expected_flux",
    [
        (0.0, 0.0, 10.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (150.0, 0.0, 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 100.0], [150.0, 50.0, 0.0, 150.0, 0.0, 0.0, 0.0, 50.0]),
        (0.0, 0.0, 100.0, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (3.0, -2.0, 0.0, [4.0, 1.0, 50.0], [11.0, 0.0, 50.0], [0.0, 0.0, 0.0, 6.0, 0.0, 1.0, 0.0, 0.0]),
        (0.0, 20.0, 0.0, [4.0, 1.0, 50.0], [0.0, 0.0, 53.75], [5.0, 1.25, 0.0, 0.0, 4.0, 0.0, 1.25, 0.0]),
        (80.0, 0.0, 10.0, [0.0, 0.0, 50.0], [0.0, 0.0, 90.0], [80.0, 30.0, 10.0, 80.0, 0.0, 0.0, 20.0, 10.0]),
    ],
)
def test_snow_soil_boundaries(
    rain: float,
    temperature: float,
    pet: float,
    state: list[float],
    expected_state: list[float],
    expected_flux: list[float],
) -> None:
    from hydrologeez.hbv_light import Forcing, Memory, Parameters, simulate

    forcing = Forcing(np.array([[[rain, temperature, pet]]]), np.ones(1), np.array([0, 1]))
    params = Parameters(
        np.array([0]), np.array([[0.0, 2.0, 0.5, 0.25, 2.0, 100.0, 2.0, 0.5]]), np.zeros((1, 5)), np.ones(1)
    )
    result = simulate(forcing, params, Memory(np.array([state]), np.zeros((1, 2)), np.zeros(0)), history=True)
    assert_allclose(result.final.zone[0], expected_state)
    assert result.history is not None
    assert_allclose(result.history.zone_flux[0, 0], expected_flux)


@pytest.mark.parametrize(
    "upper, lower, response, expected",
    [
        (1.0, 0.0, [0.0, 0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.5, 0.5, 0.0]),
        (1.0, 2.0, [5.0, 0.0, 10.0, 10.0, 2.0], [3.0, 3.0, 1.0, 0.0, 0.0, 3.0]),
        (2.0, 0.0, [0.0, 2.0, 10.0, 0.25, 0.0], [0.5, 0.5, 0.0, 0.0, 0.5, 0.0]),
    ],
)
def test_groundwater_allocation(upper: float, lower: float, response: list[float], expected: list[float]) -> None:
    from hydrologeez.hbv_light import Forcing, Memory, Parameters, simulate

    forcing = Forcing(np.zeros((1, 1, 3)), np.ones(1), np.array([0, 1]))
    params = Parameters(
        np.array([0]), np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 0.5, 1.0]]), np.array([response]), np.ones(1)
    )
    result = simulate(forcing, params, Memory(np.zeros((1, 3)), np.array([[upper, lower]]), np.zeros(0)), history=True)
    assert result.history is not None
    assert_allclose(result.history.basin_flux[0, 0], expected)


def test_shared_parameters_separate_memory_and_water_in_transit() -> None:
    from hydrologeez.hbv_light import Forcing, Memory, Parameters, simulate

    rng = np.random.default_rng(12)
    values = np.empty((60, 7, 3))
    values[..., 0] = rng.uniform(0.0, 20.0, (60, 7))
    values[..., 1] = rng.uniform(-10.0, 10.0, (60, 7))
    values[..., 2] = rng.uniform(0.0, 5.0, (60, 7))
    area = np.full(7, 1 / 7)
    forcing = Forcing(values, area, np.array([0, 7]))
    params = Parameters(
        np.array([0, 0]),
        np.tile([0.0, 2.0, 0.5, 0.25, 2.0, 100.0, 0.5, 0.5], (14, 1)),
        np.tile([1.0, 2.0, 2.0, 1.0, 0.5], (2, 1)),
        np.array([7.5, 1.0]),
    )
    initial = Memory(np.tile([4.0, 1.0, 50.0], (14, 1)), np.array([[1.0, 2.0], [1.0, 2.0]]), np.ones(7))
    result = simulate(forcing, params, initial, history=True)
    assert result.history is not None
    h = result.history
    assert not np.allclose(h.zone[:, 0], h.zone[:, 1])
    for i in range(2):
        zs, ze = result.zone_offsets[i : i + 2]
        rs, re = result.routing_offsets[i : i + 2]
        water = (
            (h.zone[:, zs:ze].sum(axis=2) * area).sum(axis=1)
            + h.groundwater[:, i].sum(axis=1)
            + h.routing[:, rs:re].sum(axis=1)
        )
        initial_water = (
            (initial.zone[zs:ze].sum(axis=1) * area).sum() + initial.groundwater[i].sum() + initial.routing[rs:re].sum()
        )
        external = ((h.zone_flux[:, zs:ze, 3] - h.zone_flux[:, zs:ze, 2]) * area).sum(axis=1)
        assert_allclose(np.diff(np.r_[initial_water, water]), external - result.outlet[:, i], atol=1e-12)
    first = simulate(forcing, params, initial, stop=31)
    tail = simulate(forcing, params, first.final, start=31)
    assert_allclose(tail.outlet, result.outlet[31:])


def test_limited_outflows_leave_restartable_nonnegative_memory() -> None:
    from hydrologeez.hbv_light import Forcing, Memory, Parameters, simulate

    forcing = Forcing(np.zeros((1, 1, 3)), np.ones(1), np.array([0, 1]))
    params = Parameters(
        np.array([0]),
        np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 1.0]]),
        np.array([[0.0, 0.0, 78.6066444633901, 12.812235153921911, 0.0]]),
        np.ones(1),
    )
    initial = Memory(np.zeros((1, 3)), np.array([[76.1142090593333, 0.0]]), np.zeros(0))
    result = simulate(forcing, params, initial)
    assert np.all(result.final.groundwater >= 0)
    simulate(forcing, params, result.final)


def test_retained_liquid_remains_restartable_at_capacity() -> None:
    from hydrologeez.hbv_light import Forcing, Memory, Parameters, simulate

    forcing = Forcing(np.array([[[2.0, 0.0, 0.0]]]), np.ones(1), np.array([0, 1]))
    params = Parameters(
        np.array([0]), np.array([[0.0, 0.0, 0.0, 0.1, 1.0, 100.0, 1.0, 1.0]]), np.zeros((1, 5)), np.ones(1)
    )
    initial = Memory(np.array([[1.0, 0.0, 0.0]]), np.zeros((1, 2)), np.zeros(0))
    result = simulate(forcing, params, initial)
    assert result.final.zone[0, 1] <= 0.1
    simulate(forcing, params, result.final)


def test_overflow_soil_remains_restartable_at_capacity() -> None:
    from hydrologeez.hbv_light import Forcing, Memory, Parameters, simulate

    forcing = Forcing(np.array([[[2.0, 0.0, 0.0]]]), np.ones(1), np.array([0, 1]))
    params = Parameters(
        np.array([0]), np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 1.0, 1.0]]), np.zeros((1, 5)), np.ones(1)
    )
    initial = Memory(np.zeros((1, 3)), np.zeros((1, 2)), np.zeros(0))
    result = simulate(forcing, params, initial)
    assert result.final.zone[0, 2] <= 0.1
    simulate(forcing, params, result.final)
