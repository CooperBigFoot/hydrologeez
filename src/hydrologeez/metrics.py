"""Discharge metrics, independent of calibration.

Pool et al. (2018), doi:10.1080/02626667.2018.1552002, equations 1–7
supply KGE and non-parametric KGE. Spearman uses average ranks for ties.
High-/low-flow biases follow the FHV/FLV signatures attributed to Yilmaz
et al. (2008), doi:10.1029/2007WR006716: independently sorted top 2% and
bottom 30%, respectively. Tail counts are rounded up. FLV compares log
shape above each tail's own minimum, not absolute low-flow volume; it is
undefined for zero flows or a constant observed tail. No floor is imposed.

Log-NSE uses log(Q + mean(observed)/100), the Pushpalatha et al. (2012)
convention documented by hydroGOF NSE. The same offset applies to both
series. All efficiency optima are 1; signed biases have optimum 0 and are
percentages. Loss takes absolute bias or 1 minus efficiency.
"""

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
OBJECTIVES = ("kge_np", "kge", "nse", "log_nse", "high_flow_bias", "low_flow_bias")
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


def evaluate(observed: Array, simulated: Array) -> dict[str, float]:
    """Score aligned 1-D discharge series; exclude only NaN observations.

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
    mask = ~np.isnan(observed)
    obs, sim = observed[mask], simulated[mask]
    result = dict.fromkeys((*OBJECTIVES, *_COMPONENTS), float("nan"))
    if obs.size < 2 or not np.all(np.isfinite(sim)) or np.any(sim < 0):
        return result
    result["nse"] = _nse(obs, sim)
    obs_mean, sim_mean = float(obs.mean()), float(sim.mean())
    if obs_mean <= 0:
        return result
    result["beta"] = beta = sim_mean / obs_mean
    result["pearson"] = pearson = _correlation(obs, sim)
    result["spearman"] = spearman = _correlation(_ranks(obs), _ranks(sim))
    obs_std = float(obs.std())
    result["alpha"] = alpha = float(sim.std()) / obs_std if obs_std > 0 else np.nan
    result["kge"] = float(1 - np.sqrt((beta - 1) ** 2 + (alpha - 1) ** 2 + (pearson - 1) ** 2))
    obs_sorted, sim_sorted = np.sort(obs), np.sort(sim)
    if sim_mean > 0:
        result["alpha_np"] = alpha_np = float(1 - 0.5 * np.sum(np.abs(sim_sorted / sim.sum() - obs_sorted / obs.sum())))
        result["kge_np"] = float(1 - np.sqrt((beta - 1) ** 2 + (alpha_np - 1) ** 2 + (spearman - 1) ** 2))
    offset = obs_mean / 100
    result["log_nse"] = _nse(np.log(obs + offset), np.log(sim + offset))
    high_count = int(np.ceil(0.02 * obs.size))
    result["high_flow_bias"] = float(100 * (sim_sorted[-high_count:].sum() / obs_sorted[-high_count:].sum() - 1))
    low_count = int(np.ceil(0.3 * obs.size))
    if obs_sorted[0] > 0 and sim_sorted[0] > 0:
        obs_shape = float(np.sum(np.log(obs_sorted[:low_count]) - np.log(obs_sorted[0])))
        sim_shape = float(np.sum(np.log(sim_sorted[:low_count]) - np.log(sim_sorted[0])))
        if obs_shape > 0:
            result["low_flow_bias"] = -100 * (sim_shape - obs_shape) / obs_shape
    return result


def objective_loss(name: str, value: float) -> float:
    """Minimized agreement loss; undefined scores are never rewarded."""
    if name not in OBJECTIVES:
        raise ValueError(f"unknown objective: {name}")
    if not np.isfinite(value):
        return float("inf")
    return abs(value) if name.endswith("_bias") else 1 - value
