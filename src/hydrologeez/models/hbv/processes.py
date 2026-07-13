"""Differentiable HBV-Light process free functions."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import ClassVar

import torch

from hydrologeez.convolution import convolve_delay_line
from hydrologeez.processes import Process

from .constants import ROUTING_BUFFER_SIZE


def partition_precipitation(
    precip: torch.Tensor, temp: torch.Tensor, tt: torch.Tensor, sfcf: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    is_rain = temp > tt
    return torch.where(is_rain, precip, 0.0), torch.where(is_rain, 0.0, sfcf * precip)


def compute_melt(temp: torch.Tensor, tt: torch.Tensor, cfmax: torch.Tensor, snow_pack: torch.Tensor) -> torch.Tensor:
    melt = torch.minimum(cfmax * (temp - tt), snow_pack)
    return torch.where(temp > tt, melt, 0.0)


def compute_refreezing(
    temp: torch.Tensor,
    tt: torch.Tensor,
    cfmax: torch.Tensor,
    cfr: torch.Tensor,
    liquid_water: torch.Tensor,
) -> torch.Tensor:
    refreeze = torch.minimum(cfr * cfmax * (tt - temp), liquid_water)
    return torch.where(temp < tt, refreeze, 0.0)


def update_snow_pack(
    sp: torch.Tensor,
    lw: torch.Tensor,
    p_snow: torch.Tensor,
    melt: torch.Tensor,
    refreeze: torch.Tensor,
    cwh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    new_sp = sp + p_snow - melt + refreeze
    new_lw_pre = lw + melt - refreeze
    lw_max = cwh * new_sp
    exceeds = new_lw_pre > lw_max
    outflow = torch.where(exceeds, new_lw_pre - lw_max, 0.0)
    new_lw = torch.where(exceeds, lw_max, new_lw_pre)
    new_sp = torch.maximum(new_sp, torch.zeros_like(new_sp))
    new_lw = torch.maximum(new_lw, torch.zeros_like(new_lw))
    return new_sp, new_lw, outflow


def compute_recharge(soil_input: torch.Tensor, sm: torch.Tensor, fc: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    safe_fc = torch.where(fc > 0.0, fc, 1.0)
    sr = torch.clamp(sm / safe_fc, 0.0, 1.0)
    sr_safe = torch.where(sr > 0.0, sr, 1.0)
    pow_term = torch.where(sr > 0.0, sr_safe**beta, 0.0)
    recharge = soil_input * pow_term
    guard = (fc <= 0.0) | (soil_input <= 0.0)
    return torch.where(guard, 0.0, recharge)


def compute_actual_et(pet: torch.Tensor, sm: torch.Tensor, fc: torch.Tensor, lp: torch.Tensor) -> torch.Tensor:
    lp_threshold = lp * fc
    safe_thr = torch.where(lp_threshold > 0.0, lp_threshold, 1.0)
    reduced = pet * sm / safe_thr
    et_act = torch.where(sm >= lp_threshold, pet, reduced)
    et_act = torch.minimum(et_act, torch.maximum(sm, torch.zeros_like(sm)))
    guard = (fc <= 0.0) | (lp <= 0.0)
    return torch.where(guard, 0.0, et_act)


def update_soil_moisture(
    sm: torch.Tensor,
    soil_input: torch.Tensor,
    recharge: torch.Tensor,
    et_act: torch.Tensor,
    fc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    new_sm_raw = sm + (soil_input - recharge) - et_act
    overflow = torch.maximum(new_sm_raw - fc, torch.zeros_like(new_sm_raw))
    return torch.clamp(new_sm_raw, min=torch.zeros_like(new_sm_raw), max=fc), overflow


def upper_zone_outflows(
    suz: torch.Tensor, k0: torch.Tensor, k1: torch.Tensor, uzl: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q0 = torch.where(suz > uzl, k0 * (suz - uzl), 0.0)
    return q0, k1 * suz


def compute_percolation(suz: torch.Tensor, perc_max: torch.Tensor) -> torch.Tensor:
    return torch.minimum(perc_max, torch.maximum(suz, torch.zeros_like(suz)))


def update_upper_zone(
    suz: torch.Tensor,
    recharge: torch.Tensor,
    q0: torch.Tensor,
    q1: torch.Tensor,
    perc: torch.Tensor,
) -> torch.Tensor:
    value = suz + recharge - q0 - q1 - perc
    return torch.maximum(value, torch.zeros_like(value))


def lower_zone_outflow(slz: torch.Tensor, k2: torch.Tensor) -> torch.Tensor:
    return k2 * slz


def update_lower_zone(slz: torch.Tensor, perc: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    value = slz + perc - q2
    return torch.maximum(value, torch.zeros_like(value))


def compute_triangular_weights(maxbas: torch.Tensor) -> torch.Tensor:
    i = torch.arange(ROUTING_BUFFER_SIZE, dtype=maxbas.dtype, device=maxbas.device)
    if maxbas.ndim:
        i = i.expand(*maxbas.shape, ROUTING_BUFFER_SIZE)
        maxbas = maxbas.unsqueeze(-1)
    t_start = i
    t_end = torch.minimum(i + 1.0, maxbas)
    half = maxbas / 2.0
    maxbas_sq = maxbas * maxbas

    r_lo = torch.minimum(t_start, half)
    r_hi = torch.minimum(t_end, half)
    rising = (r_hi * r_hi - r_lo * r_lo) / maxbas_sq

    f_lo = torch.maximum(t_start, half)
    f_hi = torch.maximum(t_end, half)
    falling = 2.0 * (f_hi - f_lo) / maxbas - (f_hi * f_hi - f_lo * f_lo) / maxbas_sq

    w_raw = rising + falling
    w = torch.where(t_end > t_start, w_raw, 0.0)
    total = torch.sum(w, dim=-1, keepdim=True)
    return torch.where(total > 0.0, w / total, w)


def convolve_routing(
    buffer: torch.Tensor, weights: torch.Tensor, qgw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return convolve_delay_line(buffer, weights, qgw)


class SnowProcess(Process):
    @abstractmethod
    def forward(
        self,
        precip: torch.Tensor,
        temp: torch.Tensor,
        snow_pack: torch.Tensor,
        liquid_water: torch.Tensor,
        parameters: Mapping[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        raise NotImplementedError


class SoilProcess(Process):
    @abstractmethod
    def forward(
        self,
        soil_input: torch.Tensor,
        pet: torch.Tensor,
        soil_moisture: torch.Tensor,
        parameters: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class ResponseProcess(Process):
    @abstractmethod
    def forward(
        self,
        upper_zone: torch.Tensor,
        lower_zone: torch.Tensor,
        recharge: torch.Tensor,
        parameters: Mapping[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        raise NotImplementedError


class RoutingProcess(Process):
    @abstractmethod
    def forward(
        self,
        routing_buffer: torch.Tensor,
        groundwater_runoff: torch.Tensor,
        parameters: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class PhysicalSnowProcess(SnowProcess):
    introduces: ClassVar[dict[str, tuple[float, float]]] = {
        "tt": (-2.5, 2.5),
        "cfmax": (0.5, 10.0),
        "sfcf": (0.4, 1.4),
        "cwh": (0.0, 0.2),
        "cfr": (0.0, 0.2),
    }

    def forward(
        self,
        precip: torch.Tensor,
        temp: torch.Tensor,
        snow_pack: torch.Tensor,
        liquid_water: torch.Tensor,
        parameters: Mapping[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        p_rain, p_snow = partition_precipitation(precip, temp, parameters["tt"], parameters["sfcf"])
        melt = compute_melt(temp, parameters["tt"], parameters["cfmax"], snow_pack)
        refreeze = compute_refreezing(
            temp,
            parameters["tt"],
            parameters["cfmax"],
            parameters["cfr"],
            liquid_water,
        )
        new_sp, new_lw, snow_outflow = update_snow_pack(
            snow_pack,
            liquid_water,
            p_snow,
            melt,
            refreeze,
            parameters["cwh"],
        )
        snow_input = p_rain + snow_outflow
        return p_rain, p_snow, new_sp, melt, new_lw, snow_input


class PhysicalSoilProcess(SoilProcess):
    introduces: ClassVar[dict[str, tuple[float, float]]] = {
        "fc": (50.0, 700.0),
        "lp": (0.3, 1.0),
        "beta": (1.0, 6.0),
    }

    def forward(
        self,
        soil_input: torch.Tensor,
        pet: torch.Tensor,
        soil_moisture: torch.Tensor,
        parameters: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recharge = compute_recharge(soil_input, soil_moisture, parameters["fc"], parameters["beta"])
        et_act = compute_actual_et(pet, soil_moisture, parameters["fc"], parameters["lp"])
        new_sm, sm_overflow = update_soil_moisture(
            soil_moisture,
            soil_input,
            recharge,
            et_act,
            parameters["fc"],
        )
        recharge_total = recharge + sm_overflow
        return new_sm, recharge_total, et_act


class PhysicalResponseProcess(ResponseProcess):
    introduces: ClassVar[dict[str, tuple[float, float]]] = {
        "k0": (0.05, 0.99),
        "k1": (0.01, 0.5),
        "k2": (0.001, 0.2),
        "perc": (0.0, 6.0),
        "uzl": (0.0, 100.0),
    }

    def forward(
        self,
        upper_zone: torch.Tensor,
        lower_zone: torch.Tensor,
        recharge: torch.Tensor,
        parameters: Mapping[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        q0, q1 = upper_zone_outflows(upper_zone, parameters["k0"], parameters["k1"], parameters["uzl"])
        perc = compute_percolation(upper_zone, parameters["perc"])
        new_suz = update_upper_zone(upper_zone, recharge, q0, q1, perc)
        q2 = lower_zone_outflow(lower_zone, parameters["k2"])
        new_slz = update_lower_zone(lower_zone, perc, q2)
        qgw = q0 + q1 + q2
        return new_suz, new_slz, q0, q1, q2, perc, qgw


class PhysicalRoutingProcess(RoutingProcess):
    introduces: ClassVar[dict[str, tuple[float, float]]] = {
        "maxbas": (1.0, 7.0),
    }

    def forward(
        self,
        routing_buffer: torch.Tensor,
        groundwater_runoff: torch.Tensor,
        parameters: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        uh_weights = compute_triangular_weights(parameters["maxbas"])
        return convolve_routing(routing_buffer, uh_weights, groundwater_runoff)
