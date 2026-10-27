"""Isolated soil temporal solve; expectations follow the documented ordered map."""

import numpy as np
from numpy.testing import assert_allclose

from experiments.hbv.temporal import affine_scan, sequential, solve


def test_affine_composition_and_worked_soil_day() -> None:
    assert_allclose(
        affine_scan(np.array([[2.0], [3.0], [0.0]]), np.array([[1.0], [4.0], [7.0]]), np.array([5.0])),
        [[11.0], [37.0], [7.0]],
    )
    inputs = np.array([[9.4]])
    pet = np.array([[2.0]])
    expected = np.array([[[52.7, 4.7, 2.0]]])
    assert_allclose(sequential(inputs, pet, np.array([50.0]), 100.0, 1.0, 0.5), expected)
    result = solve(inputs, pet, np.array([50.0]), 100.0, 1.0, 0.5)
    assert result.status == "converged"
    assert_allclose(result.values, expected)


def test_thresholds_dry_fractional_and_iteration_cap() -> None:
    inputs = np.array([[150.0], [0.0], [0.0]])
    pet = np.array([[0.0], [100.0], [0.0]])
    initial = np.array([0.0])
    expected = np.array([[[100.0, 50.0, 0.0]], [[0.0, 0.0, 100.0]], [[0.0, 0.0, 0.0]]])
    assert_allclose(sequential(inputs, pet, initial, 100.0, 0.5, 0.5), expected)
    singular = solve(inputs, pet, initial, 100.0, 0.5, 0.5)
    assert singular.status == "singular_derivative"
    assert singular.iterations == 0
    linear = solve(inputs, pet, initial, 100.0, 1.0, 0.5)
    assert linear.status == "converged"
    assert_allclose(linear.values, expected)
    capped = solve(inputs, pet, initial, 100.0, 1.0, 0.5, max_iterations=0)
    assert capped.status == "iteration_cap"
    assert capped.residual == 100.0


def test_nonlinear_derivative_and_restart() -> None:
    from experiments.hbv.temporal import derivative, soil

    m = np.array([[20.0, 70.0, 95.0]])
    inputs = np.array([[3.0, 4.0, 50.0]])
    pet = np.array([[2.0, 2.0, 2.0]])
    epsilon = 1e-4
    finite_difference = (
        soil(m + epsilon, inputs, pet, 100.0, 2.0, 0.5)[..., 0]
        - soil(m - epsilon, inputs, pet, 100.0, 2.0, 0.5)[..., 0]
    ) / (2 * epsilon)
    assert_allclose(derivative(m, inputs, pet, 100.0, 2.0, 0.5), finite_difference, atol=1e-9)
    inputs = np.full((20, 1), 3.0)
    pet = np.full_like(inputs, 2.0)
    initial = np.array([50.0])
    result = solve(inputs, pet, initial, 100.0, 2.0, 0.5)
    assert result.status == "converged"
    reference = sequential(inputs, pet, initial, 100.0, 2.0, 0.5)
    assert_allclose(result.values, reference, atol=1e-8)
    continuation = solve(inputs[10:], pet[10:], result.values[9, :, 0], 100.0, 2.0, 0.5)
    assert continuation.status == "converged"
    assert_allclose(continuation.values, reference[10:], atol=1e-8)


def test_invalid_trial_is_not_clipped() -> None:
    inputs = np.full((3, 1), 50.0)
    result = solve(inputs, np.zeros_like(inputs), np.array([0.0]), 100.0, 2.0, 0.5)
    assert result.status == "invalid_trial"
    assert result.invalid_states == 1
    assert result.iterations == 1
    assert_allclose(result.values[..., 0], 0.0)
