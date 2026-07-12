"""Torch-native differentiable hydrological performance metrics."""

import torch
from torch import Tensor

__all__ = ["kge", "lognse", "mae", "nse", "pbias", "rmse"]

LOG_EPS: float = 1e-6


def nse(obs: Tensor, sim: Tensor) -> Tensor:
    """Nash-Sutcliffe Efficiency: ``1 - SS_res / SS_tot``."""
    numerator = torch.sum(torch.square(sim - obs))
    denominator = torch.sum(torch.square(obs - torch.mean(obs)))
    return 1.0 - numerator / denominator


def rmse(obs: Tensor, sim: Tensor) -> Tensor:
    """Root Mean Squared Error."""
    return torch.sqrt(torch.mean(torch.square(sim - obs)))


def mae(obs: Tensor, sim: Tensor) -> Tensor:
    """Mean Absolute Error."""
    return torch.mean(torch.abs(sim - obs))


def pbias(obs: Tensor, sim: Tensor) -> Tensor:
    """Percent bias (hydroGOF convention): ``100 * sum(sim - obs) / sum(obs)``."""
    return 100.0 * torch.sum(sim - obs) / torch.sum(obs)


def lognse(obs: Tensor, sim: Tensor, eps: float = LOG_EPS) -> Tensor:
    """Nash-Sutcliffe Efficiency on log-transformed flows.

    Uses ``log(x + eps)`` so the transform and its gradient remain finite at zero
    flow. ``eps`` defaults to :data:`LOG_EPS`.
    """
    log_obs = torch.log(obs + eps)
    log_sim = torch.log(sim + eps)
    numerator = torch.sum(torch.square(log_sim - log_obs))
    denominator = torch.sum(torch.square(log_obs - torch.mean(log_obs)))
    return 1.0 - numerator / denominator


def kge(obs: Tensor, sim: Tensor) -> Tensor:
    """Kling-Gupta Efficiency (Gupta et al., 2009).

    ``KGE = 1 - sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)`` where ``r`` is
    the Pearson correlation, ``alpha = std(sim) / std(obs)`` is the variability
    ratio and ``beta = mean(sim) / mean(obs)`` is the bias ratio (population
    statistics, ``ddof=0``).
    """
    mean_obs = torch.mean(obs)
    mean_sim = torch.mean(sim)
    obs_anom = obs - mean_obs
    sim_anom = sim - mean_sim
    r = torch.sum(obs_anom * sim_anom) / torch.sqrt(
        torch.sum(torch.square(obs_anom)) * torch.sum(torch.square(sim_anom))
    )
    alpha = torch.std(sim, correction=0) / torch.std(obs, correction=0)
    beta = mean_sim / mean_obs
    return 1.0 - torch.sqrt(torch.square(r - 1.0) + torch.square(alpha - 1.0) + torch.square(beta - 1.0))
