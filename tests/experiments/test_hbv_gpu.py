"""Independent documented checks on real Metal, not a CPU fallback."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal GPU unavailable", allow_module_level=True)

from experiments.hbv.gpu import prepare, simulate  # noqa: E402


def test_worked_day() -> None:
    data = prepare(
        np.array([[[[6.0, 2.0, 2.0]]]]),
        np.ones((1, 1)),
        np.array([[[0.0, 2.0, 0.05, 0.1, 1.0, 100.0, 1.0, 0.5]]]),
        np.array([[1.0, 12.0, 0.2, 0.1, 0.05]]),
        np.array([[0.5, 0.5]]),
        np.array([0]),
        np.array([[[10.0, 0.0, 50.0]]]),
        np.array([[10.0, 20.0, 0.0]]),
    )
    result = simulate(data, history=True)
    np.testing.assert_allclose(np.array(result.outlet), [[1.38]], atol=3e-6)
    np.testing.assert_allclose(np.array(result.zone), [[[6.0, 0.6, 52.7]]], atol=5e-6)
    np.testing.assert_allclose(np.array(result.basin), [[11.99, 19.95, 1.38]], atol=5e-6)
    assert result.zone_history is not None
    np.testing.assert_allclose(np.array(result.zone_history)[:, 0], [[[10.0, 0.0, 50.0]]])
    np.testing.assert_allclose(np.array(result.zone_flux), [[[[9.4, 4.7, 2.0, 6.0, 4.0, 0.0, 4.7, 0.0]]]], atol=5e-6)
    np.testing.assert_allclose(np.array(result.basin_flux), [[[1.38, 2.76, 1.0, 0.34, 1.37, 1.05]]], atol=5e-6)


def test_fractional_multizone_restart_and_limits() -> None:
    forcing = np.zeros((1, 3, 2, 3))
    forcing[..., 1] = 1.0
    forcing[0, 0, :, 0] = [8.0, 2.0]
    data = prepare(
        forcing,
        np.array([[0.25, 0.75]]),
        np.broadcast_to([0.0, 2.0, 0.5, 0.0, 1.0, 10.0, 1.0, 0.5], (1, 2, 8)).copy(),
        np.array([[1.0, 1.0, 2.0, 1.0, 0.5]]),
        np.array([[0.32, 0.60, 0.08]]),
        np.array([0]),
        np.array([[[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]]]),
        np.zeros((1, 4)),
    )
    full = simulate(data, history=True)
    np.testing.assert_allclose(np.array(full.prediction), [[0.96, 1.88, 0.43]], atol=1e-6)
    assert full.basin_flux is not None
    np.testing.assert_allclose(np.array(full.basin_flux)[0, 0], [0.96, 3.0, 1.0, 15 / 11, 25 / 22, 0.5], atol=1e-6)
    first = simulate(data, stop=1)
    tail = simulate(data, start=1, initial=(first.zone, first.basin))
    np.testing.assert_array_equal(np.array(tail.prediction), np.array(full.prediction)[:, 1:])
    np.testing.assert_array_equal(np.array(tail.basin), np.array(full.basin))


@pytest.mark.parametrize(
    "rain, moisture, pet, expected",
    [(0.0, 0.0, 10.0, [0.0, 0.0, 0.0]), (150.0, 0.0, 0.0, [100.0, 50.0, 0.0]), (0.0, 1.0, 100.0, [0.0, 0.0, 1.0])],
)
def test_empty_overflow_and_evaporation_caps(rain: float, moisture: float, pet: float, expected: list[float]) -> None:
    data = prepare(
        np.array([[[[rain, 0.0, pet]]]]),
        np.ones((1, 1)),
        np.array([[[0.0, 2.0, 0.5, 0.0, 2.0, 100.0, 1.0, 0.5]]]),
        np.array([[0.0, 0.0, 1.0, 1.0, 1.0]]),
        np.ones((1, 1)),
        np.array([0]),
        np.array([[[0.0, 0.0, moisture]]]),
        np.zeros((1, 2)),
    )
    result = simulate(data, history=True)
    assert result.zone_flux is not None
    np.testing.assert_allclose(
        [np.array(result.zone)[0, 0, 2], np.array(result.prediction)[0, 0], np.array(result.zone_flux)[0, 0, 0, 2]],
        expected,
        atol=1e-6,
    )
    assert result.basin.shape == (1, 2)


def test_seeded_float32_fidelity() -> None:
    from experiments.hbv.execution import simulate as cpu_simulate
    from experiments.hbv.inputs import synthetic

    inputs = synthetic(5, 3, 400)
    data = prepare(
        inputs.forcing,
        inputs.area,
        inputs.parameters,
        inputs.response,
        inputs.routing,
        inputs.forcing_index,
        inputs.zone,
        inputs.basin,
    )
    gpu = simulate(data, history=True)
    cpu = cpu_simulate(inputs, history=True)
    for field in ("prediction", "zone", "basin", "zone_history", "basin_history", "zone_flux", "basin_flux"):
        np.testing.assert_allclose(np.array(getattr(gpu, field)), getattr(cpu, field), rtol=2e-5, atol=2e-4)
