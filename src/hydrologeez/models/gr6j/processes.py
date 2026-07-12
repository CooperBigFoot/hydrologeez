"""GR6J differentiable Torch process functions."""

from __future__ import annotations

import torch

from hydrologeez.convolution import convolve_delay_line

from .constants import EXP_BRANCH_THRESHOLD, MAX_EXP_ARG, MAX_TANH_ARG, NH, PERC_CONSTANT, D


def production_store_update(
    precip: torch.Tensor, pet: torch.Tensor, production_store: torch.Tensor, x1: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update production store and return store, ET, net and effective rain."""
    store_ratio = production_store / x1
    net_evap = pet - precip
    scaled_evap = torch.clamp_max(net_evap / x1, MAX_TANH_ARG)
    t_evap = torch.tanh(scaled_evap)
    evap_from_store = production_store * ((2.0 - store_ratio) * t_evap) / (1.0 + (1.0 - store_ratio) * t_evap)
    store_evap = production_store - evap_from_store
    et_evap = evap_from_store + precip
    net_rainfall = precip - pet
    scaled_precip = torch.clamp_max(net_rainfall / x1, MAX_TANH_ARG)
    t_precip = torch.tanh(scaled_precip)
    storage_infiltration = x1 * ((1.0 - store_ratio * store_ratio) * t_precip) / (1.0 + store_ratio * t_precip)
    effective_rainfall = net_rainfall - storage_infiltration
    store_rain = production_store + storage_infiltration
    evap_dominant = precip < pet
    new_store = torch.where(evap_dominant, store_evap, store_rain)
    actual_et = torch.where(evap_dominant, et_evap, pet)
    net_rainfall_pn = torch.where(evap_dominant, 0.0, net_rainfall)
    effective_rainfall_pr = torch.where(evap_dominant, 0.0, effective_rainfall)
    return new_store, actual_et, net_rainfall_pn, effective_rainfall_pr


def percolation(production_store: torch.Tensor, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Percolate water from the production store."""
    store = torch.clamp_min(production_store, 0.0)
    ratio4 = (store / x1) ** 4
    perc = store * (1.0 - (1.0 + ratio4 / PERC_CONSTANT) ** (-0.25))
    return store - perc, perc


def groundwater_exchange(
    routing_store: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x5: torch.Tensor
) -> torch.Tensor:
    """Compute groundwater exchange."""
    return x2 * (routing_store / x3 - x5)


def routing_store_update(
    routing_store: torch.Tensor, uh1_output: torch.Tensor, exchange: torch.Tensor, x3: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update the routing store."""
    tmp = routing_store + uh1_output + exchange
    positive = tmp >= 0.0
    actual_exchange = torch.where(positive, exchange, -(routing_store + uh1_output))
    store = torch.where(positive, tmp, 0.0)
    ratio4 = (store / x3) ** 4
    qr = torch.where(store > 0.0, store * (1.0 - (1.0 + ratio4) ** (-0.25)), 0.0)
    return store - qr, qr, actual_exchange


def exponential_store_update(
    exp_store: torch.Tensor, uh1_output: torch.Tensor, exchange: torch.Tensor, x6: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update the exponential store."""
    store = exp_store + uh1_output + exchange
    ar = torch.clamp(store / x6, -MAX_EXP_ARG, MAX_EXP_ARG)
    large_pos = store + x6 / torch.exp(ar)
    large_neg = x6 * torch.exp(ar)
    normal = x6 * torch.log1p(torch.exp(ar))
    qrexp = torch.where(
        ar > EXP_BRANCH_THRESHOLD,
        large_pos,
        torch.where(ar < -EXP_BRANCH_THRESHOLD, large_neg, normal),
    )
    return store - qrexp, qrexp


def direct_branch(uh2_output: torch.Tensor, exchange: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute direct flow and its actual exchange."""
    combined = uh2_output + exchange
    positive = combined >= 0.0
    qd = torch.where(positive, combined, 0.0)
    actual_exchange = torch.where(positive, exchange, -uh2_output)
    return qd, actual_exchange


def _ss1(i: torch.Tensor, x4: torch.Tensor) -> torch.Tensor:
    """Evaluate the UH1 S-curve."""
    below = (i / x4) ** D
    return torch.where(i <= 0.0, 0.0, torch.where(i < x4, below, 1.0))


def _ss2(i: torch.Tensor, x4: torch.Tensor) -> torch.Tensor:
    """Evaluate the UH2 S-curve."""
    ratio = i / x4
    half = 0.5 * ratio**D
    upper = 1.0 - 0.5 * torch.clamp_min(2.0 - ratio, 0.0) ** D
    return torch.where(
        i <= 0.0,
        0.0,
        torch.where(i <= x4, half, torch.where(i < 2.0 * x4, upper, 1.0)),
    )


def compute_uh_ordinates(x4: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute masked UH1 and UH2 ordinate kernels."""
    expanded_x4 = x4.unsqueeze(-1)
    i1 = torch.arange(1, NH + 1, dtype=x4.dtype, device=x4.device)
    uh1 = _ss1(i1, expanded_x4) - _ss1(i1 - 1.0, expanded_x4)
    i2 = torch.arange(1, 2 * NH + 1, dtype=x4.dtype, device=x4.device)
    uh2 = _ss2(i2, expanded_x4) - _ss2(i2 - 1.0, expanded_x4)
    return uh1, uh2


def convolve_uh(
    states: torch.Tensor, ordinates: torch.Tensor, input_value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one read-after-shift delay-line convolution step."""
    return convolve_delay_line(states, ordinates, input_value)
