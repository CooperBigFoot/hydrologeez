import numpy as np
import pytest
from numpy.testing import assert_allclose

from hydrologeez.quality_control.homogeneity import (
    Homogeneity,
    assess,
    classify,
    critical_values,
    detrend,
    homogeneity_statistics,
    prewhiten,
)


def test_statistics_match_hand_computed_step_series() -> None:
    # Mean 0.5, population SD 0.5: SNHT T(3) = 3 * 1 + 3 * 1; Buishand range 1.5 / 0.5 / sqrt(6);
    # Pettitt average ranks 2 and 5 give |2 * 6 - 3 * 7| = 9; von Neumann 1 / 1.5.
    assert_allclose(homogeneity_statistics(np.array([0.0, 0, 0, 1, 1, 1])), [6, 3 / np.sqrt(6), 9, 2 / 3])


def test_critical_values_use_published_table_with_linear_interpolation() -> None:
    n20 = np.array([9.56, 1.60, 71.0, 1.04])
    n30 = np.array([10.45, 1.70, 133.0, 1.20])
    assert_allclose(critical_values(20), n20)
    assert_allclose(critical_values(25), 0.5 * (n20 + n30))


def test_detrend_drops_subseries_with_fewer_than_three_values() -> None:
    values = np.array([1.0, 5.0, 2.0, 9.0, 3.0, np.nan])
    year = np.array([2000, 2000, 2001, 2001, 2002, 2002])
    position = np.array([0, 1, 0, 1, 0, 1])  # position 0 has 3 values; position 1 has 2
    residuals = detrend(values, year, position)
    assert_allclose(residuals[position == 0], [0.0, 0.0, 0.0], atol=1e-12)
    assert np.isnan(residuals[position == 1]).all()

    years = np.repeat(np.arange(2000, 2021), 2)
    positions = np.tile([0, 1], 21)
    series = np.where(positions == 0, np.sin(np.arange(42)), 10.0 + np.arange(42))
    reliable = (positions == 0) | (years < 2002)  # 21 reliable in position 0, 2 in position 1 -> 21 after detrend
    assessment = assess(series, reliable, years, positions)
    assert assessment.count == 21


def test_classification_maps_rejection_counts_and_numerical_errors() -> None:
    assert [classify(k) for k in range(5)] == ["useful", "useful", "doubtful", "suspect", "suspect"]
    linear = yearly(2.0 + 0.5 * np.arange(25.0), np.ones(25, dtype=np.bool_))
    assert (linear.classification, linear.count) == ("constant", 25)
    assert np.isnan(linear.statistics).all()
    overflow = yearly(np.append(np.ones(24), 1e200), np.ones(25, dtype=np.bool_))
    assert (overflow.classification, overflow.count) == ("error", 25)
    assert np.isnan(overflow.statistics).all()
    assert not overflow.rejected.any()


def yearly(values: np.ndarray, reliable: np.ndarray) -> Homogeneity:
    n = len(values)
    return assess(values, reliable, np.arange(1950, 1950 + n), np.zeros(n, dtype=np.int64))


def monthly_series(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """40 years of AR(1) noise (phi 0.3) on a seasonal cycle and trend, with 40 unreliable months."""
    rng = np.random.default_rng(seed)
    n = 480
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.3 * noise[t - 1] + rng.standard_normal()
    month = np.arange(n) % 12
    year = 1980 + np.arange(n) // 12
    values = 10 + 5 * np.sin(2 * np.pi * month / 12) + 0.15 * (year - 1980) + noise
    reliable = np.ones(n, dtype=np.bool_)
    reliable[rng.choice(n, 40, replace=False)] = False
    return values, reliable, year, month


@pytest.mark.parametrize("seed", [1, 2])
def test_monthly_shift_is_suspect_while_seasonality_trend_and_autocorrelation_are_removed(seed: int) -> None:
    values, reliable, year, month = monthly_series(seed)
    raw = homogeneity_statistics(values[reliable])
    critical = critical_values(int(reliable.sum()))
    assert (raw[:3] > critical[:3]).sum() + (raw[3] < critical[3]) >= 3  # raw tests would reject
    assert assess(values, reliable, year, month).classification == "useful"
    shifted = assess(values + 3.0 * (np.arange(len(values)) >= 160), reliable, year, month)
    assert shifted.classification == "suspect"


def test_prewhitening_estimates_lag_one_dependence_across_gaps() -> None:
    rng = np.random.default_rng(2)
    x = np.zeros(3000)
    for t in range(1, len(x)):
        x[t] = 0.6 * x[t - 1] + rng.standard_normal()
    x[rng.choice(len(x), 300, replace=False)] = np.nan
    residuals, phi = prewhiten(x)
    assert phi == pytest.approx(0.6, abs=0.03)
    assert np.isnan(residuals[np.isnan(x)]).all()
    pairs = np.isfinite(residuals[1:]) & np.isfinite(residuals[:-1])
    assert abs(np.corrcoef(residuals[1:][pairs], residuals[:-1][pairs])[0, 1]) < 0.05


def test_insufficient_and_constant_series_are_classified_without_testing() -> None:
    values = np.arange(25.0)
    reliable = np.arange(25) >= 6
    insufficient = yearly(values, reliable)
    assert (insufficient.classification, insufficient.count) == ("not sufficient data", 19)
    assert np.isnan(insufficient.statistics).all()
    constant = yearly(np.where(np.arange(30) < 5, 99.0, 4.0), np.arange(30) >= 5)
    assert (constant.classification, constant.count) == ("constant", 25)
