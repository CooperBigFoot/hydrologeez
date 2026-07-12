"""Single-zone HBV-Light as a differentiable PyTorch state-space model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from hydrologeez.models.hbv import constants, processes
from hydrologeez.models.hbv.state import HBVState
from hydrologeez.ssm import StateSpaceModel


@dataclass(frozen=True)
class HBVForcing:
    """Per-step or batched time-series HBV forcing."""

    precip: torch.Tensor
    pet: torch.Tensor
    temp: torch.Tensor


@dataclass(frozen=True)
class HBVFluxes:
    """All HBV internal fluxes for one step, in canonical 20-flux order."""

    precip: torch.Tensor
    temp: torch.Tensor
    pet: torch.Tensor
    precip_rain: torch.Tensor
    precip_snow: torch.Tensor
    snow_pack: torch.Tensor
    snow_melt: torch.Tensor
    liquid_water_in_snow: torch.Tensor
    snow_input: torch.Tensor
    soil_moisture: torch.Tensor
    recharge: torch.Tensor
    actual_et: torch.Tensor
    upper_zone: torch.Tensor
    lower_zone: torch.Tensor
    q0: torch.Tensor
    q1: torch.Tensor
    q2: torch.Tensor
    percolation: torch.Tensor
    qgw: torch.Tensor
    streamflow: torch.Tensor


class HBVModel(StateSpaceModel):
    """Single-zone, 14-parameter daily HBV-Light model."""

    n_zones = 1
    routing_buffer_size = constants.ROUTING_BUFFER_SIZE

    def __init__(
        self,
        *,
        tt: torch.Tensor,
        cfmax: torch.Tensor,
        sfcf: torch.Tensor,
        cwh: torch.Tensor,
        cfr: torch.Tensor,
        fc: torch.Tensor,
        lp: torch.Tensor,
        beta: torch.Tensor,
        k0: torch.Tensor,
        k1: torch.Tensor,
        k2: torch.Tensor,
        perc: torch.Tensor,
        uzl: torch.Tensor,
        maxbas: torch.Tensor,
    ) -> None:
        super().__init__()
        self.tt = nn.Parameter(tt)
        self.cfmax = nn.Parameter(cfmax)
        self.sfcf = nn.Parameter(sfcf)
        self.cwh = nn.Parameter(cwh)
        self.cfr = nn.Parameter(cfr)
        self.fc = nn.Parameter(fc)
        self.lp = nn.Parameter(lp)
        self.beta = nn.Parameter(beta)
        self.k0 = nn.Parameter(k0)
        self.k1 = nn.Parameter(k1)
        self.k2 = nn.Parameter(k2)
        self.perc = nn.Parameter(perc)
        self.uzl = nn.Parameter(uzl)
        self.maxbas = nn.Parameter(maxbas)

    def init_state(self, parameters: Mapping[str, torch.Tensor], *, batch_size: int) -> HBVState:
        """Initialize soil moisture to half of FC and every other store to zero."""
        zone_sm = (0.5 * parameters["fc"]).expand(batch_size)
        zeros = zone_sm.new_zeros(batch_size)
        return HBVState(
            zone_sp=zeros,
            zone_lw=zeros,
            zone_sm=zone_sm,
            upper_zone=zeros,
            lower_zone=zeros,
            routing_buffer=zone_sm.new_zeros((batch_size, self.routing_buffer_size)),
        )

    def transition(
        self,
        state: HBVState,
        forcing: HBVForcing,
        parameters: Mapping[str, torch.Tensor],
    ) -> tuple[HBVState, HBVFluxes]:
        precip = forcing.precip
        pet = forcing.pet
        temp = forcing.temp

        uh_weights = processes.compute_triangular_weights(parameters["maxbas"])

        p_rain, p_snow = processes.partition_precipitation(precip, temp, parameters["tt"], parameters["sfcf"])
        melt = processes.compute_melt(temp, parameters["tt"], parameters["cfmax"], state.zone_sp)
        refreeze = processes.compute_refreezing(
            temp,
            parameters["tt"],
            parameters["cfmax"],
            parameters["cfr"],
            state.zone_lw,
        )
        new_sp, new_lw, snow_outflow = processes.update_snow_pack(
            state.zone_sp,
            state.zone_lw,
            p_snow,
            melt,
            refreeze,
            parameters["cwh"],
        )
        snow_input = p_rain + snow_outflow

        recharge = processes.compute_recharge(
            snow_input,
            state.zone_sm,
            parameters["fc"],
            parameters["beta"],
        )
        et_act = processes.compute_actual_et(
            pet,
            state.zone_sm,
            parameters["fc"],
            parameters["lp"],
        )
        new_sm, sm_overflow = processes.update_soil_moisture(
            state.zone_sm,
            snow_input,
            recharge,
            et_act,
            parameters["fc"],
        )
        recharge_total = recharge + sm_overflow

        q0, q1 = processes.upper_zone_outflows(
            state.upper_zone,
            parameters["k0"],
            parameters["k1"],
            parameters["uzl"],
        )
        perc = processes.compute_percolation(state.upper_zone, parameters["perc"])
        new_suz = processes.update_upper_zone(state.upper_zone, recharge_total, q0, q1, perc)
        q2 = processes.lower_zone_outflow(state.lower_zone, parameters["k2"])
        new_slz = processes.update_lower_zone(state.lower_zone, perc, q2)
        qgw = q0 + q1 + q2

        qsim, new_buffer = processes.convolve_routing(state.routing_buffer, uh_weights, qgw)

        new_state = HBVState(
            zone_sp=new_sp,
            zone_lw=new_lw,
            zone_sm=new_sm,
            upper_zone=new_suz,
            lower_zone=new_slz,
            routing_buffer=new_buffer,
        )
        fluxes = HBVFluxes(
            precip=precip,
            temp=temp,
            pet=pet,
            precip_rain=p_rain,
            precip_snow=p_snow,
            snow_pack=new_sp,
            snow_melt=melt,
            liquid_water_in_snow=new_lw,
            snow_input=snow_input,
            soil_moisture=new_sm,
            recharge=recharge_total,
            actual_et=et_act,
            upper_zone=new_suz,
            lower_zone=new_slz,
            q0=q0,
            q1=q1,
            q2=q2,
            percolation=perc,
            qgw=qgw,
            streamflow=qsim,
        )
        return new_state, fluxes
