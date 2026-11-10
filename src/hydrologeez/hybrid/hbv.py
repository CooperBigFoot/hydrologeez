"""Differentiable daily HBV, with explicit dense basin/zone axes.

Forcing is [day, basin, zone, P/T/PET]. Parameters stay fixed across a run.
Zero-area zones permit batching unequal zone counts. Routing uses fixed support,
without changing the integrated triangular MAXBAS law.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Memory:
    zone: Tensor  # [basin, zone, S/L/M]
    groundwater: Tensor  # [basin, U/V]
    routing: Tensor  # [basin, future arrival lag]


@dataclass(frozen=True)
class Result:
    outlet: Tensor  # [day, basin]
    final: Memory


def positive_power(base: Tensor, exponent: Tensor) -> Tensor:
    """Exact forward zero; finite zero subgradient at empty stores.

    No smoothing: evaluate the inactive power branch at one to avoid log(0)
    in exponent derivatives. At zero, derivatives are defined as zero.
    """
    positive = base > 0
    return torch.where(positive, torch.where(positive, base, torch.ones_like(base)).pow(exponent), 0.0)


def routing_weights(maxbas: Tensor, support: int) -> Tensor:
    edges = torch.arange(support + 1, device=maxbas.device, dtype=maxbas.dtype)
    x = torch.minimum(edges[None], maxbas[:, None]) / maxbas[:, None]
    cdf = torch.where(x <= 0.5, 2 * x.square(), 1 - 2 * (1 - x).square())
    return cdf.diff(dim=-1)


def simulate(
    forcing: Tensor,
    area: Tensor,
    zone: Tensor,
    groundwater: Tensor,
    maxbas: Tensor,
    initial: Memory | None = None,
    *,
    support: int = 7,
) -> Result:
    """Evolve without mutation. Restart only with the same physical parameters.

    Default initial stores are empty; callers may supply warm-up/restart memory.
    Float64 CPU is recommended for physical reference comparisons.
    """
    if forcing.ndim != 4 or forcing.shape[-1] != 3 or forcing.shape[0] == 0:
        raise ValueError("forcing must be nonempty [day, basin, zone, P/T/PET]")
    _, basins, zones, _ = forcing.shape
    if (
        area.shape != (basins, zones)
        or zone.shape != (basins, zones, 8)
        or groundwater.shape != (basins, 4)
        or maxbas.shape != (basins,)
    ):
        raise ValueError("inconsistent basin/zone parameter shapes")
    if support < 1 or not bool(((maxbas >= 1) & (maxbas <= support)).all()):
        raise ValueError("MAXBAS outside fixed routing support")
    if not all(bool(torch.isfinite(x).all()) for x in (forcing, area, zone, groundwater, maxbas)):
        raise ValueError("physical inputs must be finite")
    if (
        bool((forcing[..., (0, 2)] < 0).any())
        or bool((area < 0).any())
        or not torch.allclose(area.sum(-1), torch.ones_like(maxbas))
    ):
        raise ValueError("nonnegative P/PET and area fractions summing to one required")
    if (
        bool((zone[..., 1:4] < 0).any())
        or bool((zone[..., 2:4] > 1).any())
        or bool((zone[..., 4:] <= 0).any())
        or bool((zone[..., 7] > 1).any())
        or bool((groundwater < 0).any())
    ):
        raise ValueError("parameters violate HBV physical domains")
    memory = initial or Memory(
        forcing.new_zeros(basins, zones, 3), forcing.new_zeros(basins, 2), forcing.new_zeros(basins, support - 1)
    )
    if (
        memory.zone.shape != (basins, zones, 3)
        or memory.groundwater.shape != (basins, 2)
        or memory.routing.shape != (basins, support - 1)
    ):
        raise ValueError("invalid restart shape")
    if (
        not all(
            bool(torch.isfinite(x).all() & (x >= 0).all()) for x in (memory.zone, memory.groundwater, memory.routing)
        )
        or bool((memory.zone[..., 2] > zone[..., 5]).any())
        or bool((memory.zone[..., 1] > zone[..., 3] * memory.zone[..., 0]).any())
    ):
        raise ValueError("invalid restart storage")
    s, liquid, moisture = memory.zone.unbind(-1)
    upper, lower = memory.groundwater.unbind(-1)
    routing = memory.routing
    tt, cfmax, cfr, cwh, scf, fc, beta, lp = zone.unbind(-1)
    perc, alpha, k1, k2 = groundwater.unbind(-1)
    weights = routing_weights(maxbas, support)
    outputs: list[Tensor] = []
    for day in forcing.unbind(0):
        p, t, pet = day.unbind(-1)
        cold = t < tt
        s = s + torch.where(cold, p * scf, 0.0)
        liquid = liquid + torch.where(cold, 0.0, p)
        melt = torch.minimum(s, cfmax * (t - tt).clamp_min(0))
        freeze = torch.minimum(liquid, cfr * cfmax * (tt - t).clamp_min(0))
        s = s + freeze - melt
        liquid = liquid + melt - freeze
        soil_input = (liquid - cwh * s).clamp_min(0)
        liquid = torch.minimum(liquid, cwh * s)
        raw = soil_input * positive_power(moisture / fc, beta)
        moisture = moisture + soil_input - raw
        overflow = (moisture - fc).clamp_min(0)
        moisture = torch.minimum(moisture, fc)
        evaporation = torch.minimum(moisture, pet * (moisture / (lp * fc)).clamp_max(1))
        moisture = moisture - evaporation
        upper = upper + ((raw + overflow) * area).sum(-1)
        d = torch.minimum(perc, upper)
        upper = upper - d
        lower = lower + d
        q1 = torch.minimum(k1 * positive_power(upper, 1 + alpha), upper)
        q2 = torch.minimum(lower, k2 * lower)
        upper, lower = upper - q1, lower - q2
        generated = q1 + q2
        outlet = weights[:, 0] * generated
        if support > 1:
            outlet = outlet + routing[:, 0]
            routing = (
                torch.cat((routing[:, 1:], torch.zeros_like(routing[:, :1])), -1) + weights[:, 1:] * generated[:, None]
            )
        outputs.append(outlet)
    return Result(
        torch.stack(outputs), Memory(torch.stack((s, liquid, moisture), -1), torch.stack((upper, lower), -1), routing)
    )
