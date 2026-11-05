"""Independent metric equations and undefined-score behavior."""

import numpy as np
import pytest

from hydrologeez.metrics import evaluate, objective_loss


def test_pool_components_with_tied_ranks() -> None:
    # Ranks [1, 2.5, 2.5, 4] and [1, 2, 3, 4]: covariance 4.5,
    # sums of rank squares 4.5 and 5. Normalized FDC distance is 0.1.
    result = evaluate(np.array([1.0, 2.0, 2.0, 5.0]), np.array([1.0, 2.0, 3.0, 4.0]))
    r = 4.5 / np.sqrt(4.5 * 5)
    assert result["beta"] == pytest.approx(1)
    assert result["alpha_np"] == pytest.approx(0.9)
    assert result["spearman"] == pytest.approx(r)
    assert result["kge_np"] == pytest.approx(1 - np.sqrt(0.1**2 + (r - 1) ** 2))


def test_scale_bias_and_original_efficiencies() -> None:
    result = evaluate(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
    assert result["kge_np"] == pytest.approx(0)
    assert result["alpha_np"] == pytest.approx(1)
    assert result["kge"] == pytest.approx(1 - np.sqrt(2))
    assert result["nse"] == pytest.approx(-6)
    assert result["high_flow_bias"] == pytest.approx(100)
    assert objective_loss("high_flow_bias", -100) == objective_loss("high_flow_bias", 100) == 100
    assert objective_loss("kge_np", 0.75) == pytest.approx(0.25)


def test_log_nse_zero_offset_and_unit_invariance() -> None:
    obs = np.array([0.0, 1.0, 2.0])
    sim = np.array([0.0, 2.0, 1.0])
    # mean observed is 1, hence the independently specified offset is .01.
    logs = np.log(np.array([0.01, 1.01, 2.01]))
    expected = 1 - 2 * (logs[2] - logs[1]) ** 2 / np.sum((logs - logs.mean()) ** 2)
    assert evaluate(obs, sim)["log_nse"] == pytest.approx(expected)
    assert evaluate(obs * 1000, sim * 1000)["log_nse"] == pytest.approx(expected)


def test_fdc_tails_are_independently_sorted_and_flv_measures_shape() -> None:
    obs = np.arange(1.0, 11.0)
    sim = np.array([12.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 0.5])
    result = evaluate(obs, sim)
    assert result["high_flow_bias"] == pytest.approx(20)
    # Bottom three observed [1,2,3], simulated [.5,2,3].
    expected = -100 * (np.log(24) - np.log(6)) / np.log(6)
    assert result["low_flow_bias"] == pytest.approx(expected)
    assert evaluate(obs, 2 * obs)["low_flow_bias"] == pytest.approx(0)


def test_missing_observations_do_not_drop_candidate_failures() -> None:
    obs = np.array([1.0, np.nan, 2.0, 3.0])
    perfect = evaluate(obs, np.array([1.0, np.inf, 2.0, 3.0]))
    assert perfect["kge_np"] == pytest.approx(1)
    failed = evaluate(obs, np.array([1.0, 0.0, np.nan, 3.0]))
    assert all(np.isnan(value) for value in failed.values())
    assert objective_loss("kge_np", failed["kge_np"]) == np.inf


@pytest.mark.parametrize("obs,sim", [([0.0, 0.0], [0.0, 0.0]), ([1.0, 1.0], [1.0, 1.0]), ([1.0, 2.0], [0.0, 0.0])])
def test_degenerate_correlation_never_becomes_perfect(obs: list[float], sim: list[float]) -> None:
    result = evaluate(np.array(obs), np.array(sim))
    assert np.isnan(result["kge_np"])
    assert objective_loss("kge_np", result["kge_np"]) == np.inf


def test_flv_zero_or_constant_observed_tail_is_undefined() -> None:
    for obs in [np.arange(10.0), np.array([1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])]:
        assert np.isnan(evaluate(obs, obs)["low_flow_bias"])


def test_metric_input_errors_and_empty_scores() -> None:
    with pytest.raises(ValueError, match="aligned"):
        evaluate(np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="nonnegative"):
        evaluate(np.array([-1.0, 1.0]), np.ones(2))
    assert all(np.isnan(value) for value in evaluate(np.array([]), np.array([])).values())
    with pytest.raises(ValueError, match="unknown objective"):
        objective_loss("beta", 1.0)
