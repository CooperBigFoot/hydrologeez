"""Deterministic discharge metrics; numerical conventions are documented in docs/metrics.md."""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

Array = NDArray[np.float64]
OBJECTIVES = (
    "kge_np",
    "kge",
    "nse",
    "log_nse",
    "high_flow_bias",
    "low_flow_bias",
    "mse",
    "rmse",
    "alpha",
    "beta",
    "beta_nse",
    "pearson",
    "middle_flow_bias",
    "peak_timing",
    "missed_peaks",
    "peak_mape",
)
_COMPONENTS = ("beta", "alpha_np", "spearman", "alpha", "pearson")


def _correlation(observed: Array, simulated: Array) -> float:
    obs = observed - observed.mean()
    sim = simulated - simulated.mean()
    denominator = float(np.sqrt(np.dot(obs, obs) * np.dot(sim, sim)))
    return float(np.clip(np.dot(obs, sim) / denominator, -1, 1)) if denominator > 0 else np.nan


def _ranks(values: Array) -> Array:
    order = np.argsort(values)
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
    ends = np.r_[starts[1:], values.size]
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.repeat((starts + ends - 1) / 2, ends - starts)
    return ranks


def _nse(observed: Array, simulated: Array) -> float:
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    return float(1 - np.sum((simulated - observed) ** 2) / denominator) if denominator > 0 else np.nan


def _peak_scores(obs: Array, sim: Array, days: NDArray[np.int64]) -> dict[str, float]:
    peaks, _ = find_peaks(obs, distance=100, prominence=float(obs.std()))
    timing: list[float] = []
    for idx in peaks:
        if idx < 3 or idx + 3 >= obs.size or days[idx + 3] - days[idx - 3] != 6:
            continue
        matched = (
            int(idx)
            if sim[idx] > sim[idx - 1] and sim[idx] > sim[idx + 1]
            else int(idx - 3 + np.argmax(sim[idx - 3 : idx + 4]))
        )
        timing.append(float(abs(days[idx] - days[matched])))
    observed_peaks, _ = find_peaks(obs, distance=30, height=float(np.percentile(obs, 80)))
    simulated_peaks, _ = find_peaks(sim, distance=30, height=float(np.percentile(sim, 80)))
    missed = 0
    for idx in observed_peaks:
        if idx < 1 or idx + 1 >= obs.size or days[idx + 1] - days[idx - 1] != 2:
            continue
        if not np.any(np.abs(simulated_peaks - idx) <= 1):
            missed += 1
    return {
        "peak_timing": float(np.mean(timing)) if timing else np.nan,
        "missed_peaks": missed / len(observed_peaks) if len(observed_peaks) else 0.0,
        "peak_mape": float(100 * np.mean(np.abs((sim[peaks] - obs[peaks]) / obs[peaks]))) if len(peaks) else np.nan,
    }


def evaluate(observed: Array, simulated: Array, *, dates: NDArray[np.datetime64] | None = None) -> dict[str, float]:
    """Score aligned 1-D discharge series; exclude only NaN observations.

    Without dates, array positions are consecutive daily observations. Explicit
    dates must be increasing whole days; missing-day gaps exclude peak windows.
    Missing observations define a candidate-independent mask. Nonfinite or
    negative scored simulations invalidate all scores; infinities/negative
    observations are errors, not missing data. Undefined individual metrics
    return NaN, including correlation for constant series. Inputs are not
    mutated. At least two observed days are required.
    """
    observed = np.asarray(observed, dtype=np.float64)
    simulated = np.asarray(simulated, dtype=np.float64)
    if observed.ndim != 1 or simulated.shape != observed.shape:
        raise ValueError("observed and simulated must be aligned one-dimensional arrays")
    if np.any(np.isinf(observed)) or np.any(observed < 0):
        raise ValueError("observations must be nonnegative finite values or NaN")
    if dates is None:
        days = np.arange(observed.size)
    else:
        dates = np.asarray(dates)
        if dates.shape != observed.shape or not np.issubdtype(dates.dtype, np.datetime64):
            raise ValueError("dates must be aligned datetime64 daily coordinates")
        daily = dates.astype("datetime64[D]")
        if np.any(np.isnat(dates)) or np.any(dates != daily) or np.any(np.diff(daily) <= np.timedelta64(0, "D")):
            raise ValueError("dates must be strictly increasing whole days without NaT")
        days = daily.astype(np.int64)
    mask = ~np.isnan(observed)
    obs, sim = observed[mask], simulated[mask]
    result = dict.fromkeys((*OBJECTIVES, *_COMPONENTS), float("nan"))
    if obs.size < 2 or not np.all(np.isfinite(sim)) or np.any(sim < 0):
        return result
    result["nse"] = _nse(obs, sim)
    obs_mean, sim_mean = float(obs.mean()), float(sim.mean())
    result["mse"] = float(np.mean((sim - obs) ** 2))
    result["rmse"] = float(np.sqrt(result["mse"]))
    result.update(_peak_scores(obs, sim, days[mask]))
    result["beta"] = beta = sim_mean / obs_mean if obs_mean > 0 else np.nan
    result["pearson"] = pearson = _correlation(obs, sim)
    result["spearman"] = spearman = _correlation(_ranks(obs), _ranks(sim))
    obs_std = float(obs.std())
    result["alpha"] = alpha = float(sim.std()) / obs_std if obs_std > 0 else np.nan
    result["beta_nse"] = (sim_mean - obs_mean) / obs_std if obs_std > 0 else np.nan
    result["kge"] = float(1 - np.sqrt((beta - 1) ** 2 + (alpha - 1) ** 2 + (pearson - 1) ** 2))
    obs_sorted, sim_sorted = np.sort(obs), np.sort(sim)
    if sim_mean > 0 and obs_mean > 0:
        result["alpha_np"] = alpha_np = float(1 - 0.5 * np.sum(np.abs(sim_sorted / sim.sum() - obs_sorted / obs.sum())))
        result["kge_np"] = float(1 - np.sqrt((beta - 1) ** 2 + (alpha_np - 1) ** 2 + (spearman - 1) ** 2))
    offset = obs_mean / 100
    if offset > 0:
        result["log_nse"] = _nse(np.log(obs + offset), np.log(sim + offset))
    high_count = int(np.round(0.02 * obs.size))
    obs_high, sim_high = obs_sorted[::-1][:high_count], sim_sorted[::-1][:high_count]
    if obs_high.sum() > 0:
        result["high_flow_bias"] = float(100 * np.sum(sim_high - obs_high) / obs_high.sum())
    lower, upper = int(np.round(0.2 * obs.size)), int(np.round(0.7 * obs.size))
    obs_logs = np.log(np.where(obs_sorted == 0, 1e-6, obs_sorted)[::-1])
    sim_logs = np.log(np.where(sim_sorted == 0, 1e-6, sim_sorted)[::-1])
    if upper < obs.size:
        obs_slope = obs_logs[lower] - obs_logs[upper]
        sim_slope = sim_logs[lower] - sim_logs[upper]
        result["middle_flow_bias"] = float(100 * (sim_slope - obs_slope) / (obs_slope + 1e-6))
    low_count = int(np.round(0.3 * obs.size))
    obs_log, sim_log = obs_logs[-low_count:], sim_logs[-low_count:]
    obs_shape = float(np.sum(obs_log - obs_log.min()))
    sim_shape = float(np.sum(sim_log - sim_log.min()))
    result["low_flow_bias"] = -100 * (sim_shape - obs_shape) / (obs_shape + 1e-6)
    return result


def objective_loss(name: str, value: float) -> float:
    """Minimized agreement loss; undefined scores are never rewarded."""
    if name not in OBJECTIVES:
        raise ValueError(f"unknown objective: {name}")
    if not np.isfinite(value):
        return float("inf")
    if name in ("alpha", "beta"):
        return abs(value - 1)
    if name.endswith("_bias") or name == "beta_nse":
        return abs(value)
    if name in ("mse", "rmse", "peak_timing", "missed_peaks", "peak_mape"):
        return value
    return 1 - value
