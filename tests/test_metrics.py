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
    assert np.isnan(result["high_flow_bias"])  # round(0.02 * 3) selects no high flows.
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
    assert evaluate(np.tile(obs, 10), np.tile(sim, 10))["high_flow_bias"] == pytest.approx(20)
    # Bottom three observed [1,2,3], simulated [.5,2,3].
    expected = -100 * (np.log(24) - np.log(6)) / (np.log(6) + 1e-6)
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


@pytest.mark.parametrize(
    "obs_tail,sim_tail",
    [
        ([0, 1, 2], [1, 1, 2]),
        ([1, 1, 2], [0, 1, 2]),
        ([0, 1, 2], [0, 1, 2]),
        ([0, 0, 0], [0, 0, 0]),
        ([1, 1, 1], [1, 1, 2]),
    ],
)
def test_flv_regularized_tails(obs_tail: list[float], sim_tail: list[float]) -> None:
    obs = np.array([*obs_tail, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    sim = np.array([*sim_tail, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    # Independently worked three-point shapes: zero is replaced by 1e-6.
    shapes: dict[tuple[float, ...], float] = {
        (0, 1, 2): np.log(1e6) + np.log(2e6),
        (1, 1, 2): np.log(2),
        (0, 0, 0): 0,
        (1, 1, 1): 0,
    }
    observed_shape, simulated_shape = shapes[tuple(obs_tail)], shapes[tuple(sim_tail)]
    expected = -100 * (simulated_shape - observed_shape) / (observed_shape + 1e-6)
    assert evaluate(obs, sim)["low_flow_bias"] == pytest.approx(expected)


def test_metric_input_errors_and_empty_scores() -> None:
    with pytest.raises(ValueError, match="aligned"):
        evaluate(np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="nonnegative"):
        evaluate(np.array([-1.0, 1.0]), np.ones(2))
    assert all(np.isnan(value) for value in evaluate(np.array([]), np.array([])).values())
    with pytest.raises(ValueError, match="unknown objective"):
        objective_loss("unknown", 1.0)


@pytest.mark.parametrize(
    "name,value,loss",
    [
        ("mse", 4, 4),
        ("rmse", 2, 2),
        ("alpha", 2, 1),
        ("beta", 0.5, 0.5),
        ("beta_nse", -2, 2),
        ("pearson", 0.7, 0.3),
        ("middle_flow_bias", -20, 20),
        ("peak_timing", 2, 2),
        ("missed_peaks", 0.4, 0.4),
        ("peak_mape", 50, 50),
    ],
)
def test_objective_optima(name: str, value: float, loss: float) -> None:
    assert objective_loss(name, value) == pytest.approx(loss)
    assert objective_loss(name, np.nan) == np.inf


def test_error_metrics_and_components() -> None:
    result = evaluate(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
    assert result["mse"] == pytest.approx(14 / 3)
    assert result["rmse"] == pytest.approx(np.sqrt(14 / 3))
    assert result["beta_nse"] == pytest.approx(2 / np.sqrt(2 / 3))


def test_peak_matching_and_missing_day_gap() -> None:
    obs = np.ones(201)
    sim = np.ones(201)
    obs[100] = 10
    sim[102] = 5
    result = evaluate(obs, sim)
    assert result["peak_timing"] == 2
    assert result["missed_peaks"] == 1
    assert result["peak_mape"] == 90
    # A local simulated peak at the observed peak takes priority over a larger neighbor.
    sim[100] = 2
    assert evaluate(obs, sim)["peak_timing"] == 0
    obs[99] = np.nan
    assert np.isnan(evaluate(obs, sim)["peak_timing"])
    dates = np.arange("2000-01-01", "2000-07-20", dtype="datetime64[D]")
    assert np.isnan(evaluate(obs, sim, dates=dates)["peak_timing"])


def test_rounded_high_tail_and_zero_series() -> None:
    assert np.isnan(evaluate(np.arange(1.0, 11.0), np.arange(1.0, 11.0))["high_flow_bias"])
    scores = evaluate(np.zeros(10), np.zeros(10))
    assert scores["mse"] == scores["rmse"] == scores["low_flow_bias"] == 0
    assert scores["missed_peaks"] == 0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0])
def test_invalid_simulation_cannot_improve_any_score(bad: float) -> None:
    obs = np.array([0.0, 1.0, 2.0])
    sim = np.array([bad, 1.0, 2.0])
    assert all(np.isnan(value) for value in evaluate(obs, sim).values())


@pytest.mark.parametrize(
    "dates",
    [
        np.array(["2000-01-01", "2000-01-01"], dtype="datetime64[D]"),
        np.array(["NaT", "2000-01-02"], dtype="datetime64[D]"),
        np.array(["2000-01-01T12", "2000-01-02T12"], dtype="datetime64[h]"),
    ],
)
def test_invalid_daily_coordinates(dates: np.ndarray) -> None:
    with pytest.raises(ValueError, match="whole days"):
        evaluate(np.ones(2), np.ones(2), dates=dates)


def test_missing_peak_windows_stay_in_missed_peak_denominator() -> None:
    obs = np.ones(201)
    obs[100] = 10
    obs[99] = np.nan
    scores = evaluate(obs, np.ones(201))
    assert scores["missed_peaks"] == 0  # Reference skips this numerator, not its denominator.
    assert scores["peak_mape"] == 90


def test_flv_zero_replacement_is_not_a_positive_flow_floor() -> None:
    obs = np.array([0.0, 1e-8, 1e-7, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    sim = np.array([0.0, 1e-8, 1e-6, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    # Selected [1e-7, 1e-8, 1e-6] anchors at 1e-8, not its last element.
    assert evaluate(obs, sim)["low_flow_bias"] == pytest.approx(-100 * np.log(10) / (np.log(1000) + 1e-6))
