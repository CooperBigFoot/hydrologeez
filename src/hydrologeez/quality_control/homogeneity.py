"""Homogeneity assessment of index series (GSIM Part 2, Sect. 4; ECA&D ATBD 2013, Sect. 8).

Unreliable or undefined index values are set to missing. With fewer than 20 remaining values
the series is "not sufficient data"; if they are all equal it is "constant". Otherwise a linear
trend and mean (x = b + a t, t in years) is removed by least squares from the whole yearly
series, or separately from each season or month. The result is pre-whitened with a lag-1
autoregressive model fitted by exact Gaussian maximum likelihood, missing values are removed,
and the SNHT, Buishand range, Pettitt and von Neumann ratio tests are applied at the 1% level
to the scale-invariant test statistics. Zero or one rejection is "useful", two "doubtful",
three or four "suspect"; a numerical failure is "error".

Conventions where the paper is silent: a season or month subseries with fewer than 3 reliable
values has no residual degrees of freedom under a 2-parameter line and is set to missing after
detrending; if fewer than 20 values remain it is "not sufficient data", and if the detrended
residuals are all equal within roundoff it is "constant". The detrended series has zero mean,
so the AR(1) model has no intercept. Across a gap of k periods the innovation is
x_t - phi^k x_{t-k}, and residuals are innovations scaled to the common innovation variance
(Kalman-filter standardization, as in R's ``arima``). Critical values interpolate linearly in n
between the published rows (n = 20...100) and simulated rows beyond (see
``SIMULATED_CRITICAL_VALUES``), which is mildly conservative on the convex Pettitt curve and
clamps above n = 5000.

A rejected test is statistical evidence of a change, not proof of measurement error: real
regulation or land-use change can alter a river's behaviour.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata

from hydrologeez.quality_control.indices import IndexSeries

type Classification = Literal["useful", "doubtful", "suspect", "not sufficient data", "constant", "error"]

CLASSIFICATIONS: tuple[Classification, ...] = (
    "useful",
    "doubtful",
    "suspect",
    "not sufficient data",
    "constant",
    "error",
)
TESTS = ("snht", "buishand", "pettitt", "von_neumann")
MINIMUM_VALUES = 20

# ECA&D ATBD (2013) Tables 3-6: n, SNHT T0, Buishand R/sqrt(n), Pettitt XE, von Neumann N (lower tail).
PUBLISHED_CRITICAL_VALUES = np.array(
    [
        (20, 9.56, 1.60, 71, 1.04),
        (30, 10.45, 1.70, 133, 1.20),
        (40, 11.01, 1.74, 208, 1.29),
        (50, 11.38, 1.78, 293, 1.36),
        (70, 11.89, 1.81, 488, 1.45),
        (100, 12.32, 1.86, 841, 1.54),
    ]
)

# Extension for n > 100 from scripts/homogeneity_critical_values.py (100 000 iid normal series per n,
# seed 20180417); it reproduces the published rows above to within simulation error.
SIMULATED_CRITICAL_VALUES = np.array(
    [
        (150, 12.75, 1.877, 1549, 1.625),
        (200, 13.02, 1.906, 2411, 1.672),
        (300, 13.35, 1.920, 4453, 1.733),
        (400, 13.46, 1.931, 6870, 1.767),
        (500, 13.56, 1.937, 9639, 1.793),
        (700, 13.79, 1.949, 15979, 1.825),
        (1000, 14.10, 1.956, 27406, 1.853),
        (1500, 14.29, 1.971, 50343, 1.881),
        (2000, 14.46, 1.980, 77588, 1.896),
        (3000, 14.61, 1.973, 142431, 1.916),
        (5000, 14.84, 1.992, 309900, 1.934),
    ]
)
CRITICAL_VALUES = np.vstack([PUBLISHED_CRITICAL_VALUES, SIMULATED_CRITICAL_VALUES])
_PHI_BOUND = 1 - 1e-9


@dataclass(frozen=True)
class Homogeneity:
    """Assessment of one index series; statistics and rejections follow TESTS (NaN/False if untested)."""

    classification: Classification
    count: int
    statistics: NDArray[np.float64]
    rejected: NDArray[np.bool_]


def homogeneity_statistics(series: NDArray[np.float64]) -> NDArray[np.float64]:
    """SNHT T0, Buishand R/sqrt(n), Pettitt XE and von Neumann N along the last axis.

    SNHT and Buishand standardize by the population SD (sum of squares / n), which makes T0
    consistent with Jaruskova's two-sample relation quoted in the ATBD. Pettitt ranks ties
    by their average rank.
    """
    n = series.shape[-1]
    centred = series - series.mean(axis=-1, keepdims=True)
    squares = (centred**2).sum(axis=-1)
    partial = np.cumsum(centred, axis=-1)
    k = np.arange(1, n)
    snht = (partial[..., :-1] ** 2 * n / (k * (n - k))).max(axis=-1) / (squares / n)
    bounded = np.concatenate([np.zeros((*series.shape[:-1], 1)), partial], axis=-1)
    buishand = (bounded.max(axis=-1) - bounded.min(axis=-1)) / np.sqrt(squares / n) / np.sqrt(n)
    ranks = rankdata(series, axis=-1)
    pettitt = np.abs(2 * np.cumsum(ranks, axis=-1) - np.arange(1, n + 1) * (n + 1)).max(axis=-1)
    von_neumann = (np.diff(series, axis=-1) ** 2).sum(axis=-1) / squares
    return np.stack([snht, buishand, pettitt, von_neumann], axis=-1)


def critical_values(n: int) -> NDArray[np.float64]:
    """1% critical values in TESTS order; reject above them, or below for von Neumann."""
    return np.array([np.interp(n, CRITICAL_VALUES[:, 0], CRITICAL_VALUES[:, column]) for column in range(1, 5)])


def classify(rejections: int) -> Classification:
    """Map the number of rejected tests (0..4) to its ECA&D / GSIM class."""
    return CLASSIFICATIONS[min(max(rejections - 1, 0), 2)]


def detrend(values: NDArray[np.float64], year: NDArray[np.int64], position: NDArray[np.int64]) -> NDArray[np.float64]:
    """Remove a least-squares line in years from each season/month subseries with >= 3 values."""
    residuals = np.full(len(values), np.nan)
    for p in np.unique(position):
        members = np.flatnonzero((position == p) & np.isfinite(values))
        if len(members) < 3:
            continue
        t = year[members] - year[members].mean()
        design = np.column_stack([np.ones(len(members)), t])
        coefficients = np.linalg.lstsq(design, values[members], rcond=None)[0]
        residuals[members] = values[members] - design @ coefficients
    return residuals


def prewhiten(values: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    """Residuals of a zero-mean AR(1) fitted by exact maximum likelihood on consecutive periods.

    NaN marks missing periods; their residuals stay NaN. Failed optimization yields NaN.
    """
    observed = np.flatnonzero(np.isfinite(values))
    x = values[observed]
    gaps = np.diff(observed)

    def innovations(phi: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lagged = phi**gaps
        gain = np.concatenate([[1 / (1 - phi**2)], (1 - lagged**2) / (1 - phi**2)])
        return np.concatenate([x[:1], x[1:] - lagged * x[:-1]]), gain

    def deviance(phi: float) -> float:
        error, gain = innovations(phi)
        return float(len(x) * np.log(np.sum(error**2 / gain) / len(x)) + np.sum(np.log(gain)))

    fit = minimize_scalar(deviance, bounds=(-_PHI_BOUND, _PHI_BOUND), method="bounded", options={"xatol": 1e-10})
    residuals = np.full(len(values), np.nan)
    if not fit.success:
        return residuals, float("nan")
    error, gain = innovations(float(fit.x))
    residuals[observed] = error / np.sqrt(gain)
    return residuals, float(fit.x)


def assess(
    values: NDArray[np.float64], reliable: NDArray[np.bool_], year: NDArray[np.int64], position: NDArray[np.int64]
) -> Homogeneity:
    """Assess one index series on consecutive periods with their year and season/month position."""
    kept = reliable & np.isfinite(values)
    count = int(kept.sum())
    untested = np.full(len(TESTS), np.nan), np.zeros(len(TESTS), dtype=np.bool_)
    if count < MINIMUM_VALUES:
        return Homogeneity("not sufficient data", count, *untested)
    scale = float(np.ptp(values[kept]))
    if scale == 0:
        return Homogeneity("constant", count, *untested)
    detrended = detrend(np.where(kept, values, np.nan), year, position)
    kept = np.isfinite(detrended)
    count = int(kept.sum())
    if count < MINIMUM_VALUES:
        return Homogeneity("not sufficient data", count, *untested)
    if np.ptp(detrended[kept]) <= 1e-12 * scale:
        return Homogeneity("constant", count, *untested)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        residuals, _ = prewhiten(detrended)
        statistics = homogeneity_statistics(residuals[kept])
    if not np.isfinite(statistics).all():
        return Homogeneity("error", count, *untested)
    critical = critical_values(count)
    rejected = np.append(statistics[:3] > critical[:3], statistics[3] < critical[3])
    return Homogeneity(classify(int(rejected.sum())), count, statistics, rejected)


def assess_indices(series: IndexSeries) -> list[Homogeneity]:
    """Assess every index of a series, in ``series.names`` order."""
    return [assess(column, series.reliable, series.year, series.position) for column in series.values.T]
