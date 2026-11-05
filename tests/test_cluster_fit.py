import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from hydrologeez.cluster.fit import distances, fit_candidates, pooled_scale


def test_pooled_scaling_preserves_between_basin_offset() -> None:
    values = np.array([[[0.0, 10.0], [2.0, 10.0]], [[4.0, 20.0], [6.0, 20.0]]])
    scaled, mean, std = pooled_scale(values)
    assert_allclose(mean, [3, 15])
    assert_allclose(std, [np.sqrt(5), 5])
    assert scaled[1, :, 0].mean() > scaled[0, :, 0].mean()
    assert_allclose(scaled * std + mean, values)


def test_dependent_distance_and_band() -> None:
    # Opposite channel shifts cannot both be aligned at zero cost on one path.
    a = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    b = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    assert distances(a[None], b[None], radius=2)[0, 0] > 0
    assert_allclose(distances(a[None], b[None], radius=0), [[2.0]])


def test_centroid_and_objective_obey_explicit_zero_radius() -> None:
    # With no warping, the centroid is the pointwise mean and each squared
    # distance is 0.5. Radius zero discriminates the bug, not the study tolerance.
    values = np.array([[[0.0], [0.0], [1.0], [0.0], [0.0]], [[0.0], [1.0], [0.0], [0.0], [0.0]]])
    fit = fit_candidates(values, (1,), seed=3, n_init=1, max_iter=10, tolerance=1e-6, radius=0)[1]
    assert_allclose(fit.centroids, [[[0.0], [0.5], [0.5], [0.0], [0.0]]])
    assert_allclose(distances(values, fit.centroids, radius=0), np.full((2, 1), np.sqrt(0.5)))
    assert_allclose(fit.inertia, 0.5)
    assert_array_equal(fit.labels, [0, 0])


def test_fit_records_measured_convergence_and_reproducibility() -> None:
    values = np.array([np.full((4, 2), x) for x in (0.0, 0.1, 5.0, 5.1)])
    result = fit_candidates(values, (2,), seed=3, n_init=2, max_iter=10, tolerance=1e-6, radius=2)
    repeat = fit_candidates(values, (2,), seed=3, n_init=2, max_iter=10, tolerance=1e-6, radius=2)
    fit = result[2]
    assert_array_equal(fit.labels, repeat[2].labels)
    assert fit.converged
    assert fit.history[-1] == fit.history[-2]
    assert fit.silhouette is not None and fit.silhouette > 0.95
    assert_array_equal(fit.labels, distances(values, fit.centroids, radius=2).argmin(axis=1))
    assert_allclose(fit.inertia, np.mean(distances(values, fit.centroids, radius=2).min(axis=1) ** 2))


def test_band_prevents_three_step_shift_and_barycenter_is_not_mean() -> None:
    from tslearn.barycenters import dtw_barycenter_averaging_petitjean

    early = np.zeros((8, 2))
    late = early.copy()
    early[2] = [4, 8]
    late[5] = [4, 8]
    assert_allclose(distances(early[None], late[None], radius=3), [[0.0]])
    assert distances(early[None], late[None], radius=2)[0, 0] > 0
    late = np.roll(early, 1, axis=0)
    values = np.stack([early, late])
    expected = dtw_barycenter_averaging_petitjean(
        values, init_barycenter=early, metric_params={"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": 2}
    )
    assert not np.allclose(expected, values.mean(axis=0))
    assert_allclose(expected.max(axis=0), [4, 8])


def test_convergence_on_final_iteration_is_measured() -> None:
    values = np.array([np.full((4, 2), x) for x in (0.0, 0.1, 5.0, 5.1)])
    full = fit_candidates(values, (2,), seed=3, n_init=1, max_iter=10, tolerance=1e-6, radius=2)[2]
    last = fit_candidates(values, (2,), seed=3, n_init=1, max_iter=len(full.history), tolerance=1e-6, radius=2)[2]
    assert last.converged
    assert last.history == full.history
    limited = fit_candidates(values, (2,), seed=3, n_init=1, max_iter=1, tolerance=1e-6, radius=2)[2]
    assert not limited.converged
    assert len(limited.history) == 1


def test_identical_curves_cannot_silently_reduce_requested_count() -> None:
    import pytest

    with pytest.raises(ValueError, match="no successful fit"):
        fit_candidates(np.zeros((3, 4, 2)), (2,), seed=1, n_init=1, max_iter=3, tolerance=1e-6, radius=2)
