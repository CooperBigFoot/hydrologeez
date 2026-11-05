import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from hydrologeez.calibration._evolution import evolve, rank_and_crowding


def test_front_rank_and_crowding_independent_example() -> None:
    values = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0], [4.0, 4.0], [np.inf, 0.0]])
    rank, distance = rank_and_crowding(values)
    assert_array_equal(rank, [0, 0, 0, 0, 1, 2])
    assert np.isinf(distance[[0, 3]]).all()
    assert_allclose(distance[[1, 2]], [4 / 3, 4 / 3])


def test_seeded_evolution_improves_sphere_and_respects_bounds() -> None:
    bounds = np.array([[-2.0, 4.0], [-3.0, 1.0]])

    def objective(x: np.ndarray) -> np.ndarray:
        return np.sum(x * x, axis=1)[:, None]

    initial = evolve(objective, bounds, 20, 0, 71)
    first = evolve(objective, bounds, 20, 30, 71)
    second = evolve(objective, bounds, 20, 30, 71)
    assert_array_equal(first.parameters, second.parameters)
    assert first.losses.min() < initial.losses.min()
    assert np.all(first.parameters >= bounds[:, 0])
    assert np.all(first.parameters <= bounds[:, 1])


def test_multiobjective_tradeoff_survives_and_invalid_is_last() -> None:
    def objective(x: np.ndarray) -> np.ndarray:
        return np.column_stack((x[:, 0] ** 2, (1 - x[:, 0]) ** 2))

    result = evolve(objective, np.array([[0.0, 1.0]]), 20, 10, 2)
    assert np.all(result.rank == 0)
    assert result.parameters.min() < 0.1
    assert result.parameters.max() > 0.9
    rank, crowding = rank_and_crowding(np.array([[np.inf, 0.0], [1.0, 2.0], [2.0, 1.0]]))
    assert_array_equal(rank, [1, 0, 0])
    assert crowding[0] == 0
