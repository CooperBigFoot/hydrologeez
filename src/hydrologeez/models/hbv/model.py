"""Single-zone HBV-Light as a differentiable PyTorch state-space model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from hydrologeez.models.hbv import constants
from hydrologeez.models.hbv.processes import (
    PhysicalResponseProcess,
    PhysicalRoutingProcess,
    PhysicalSnowProcess,
    PhysicalSoilProcess,
    ResponseProcess,
    RoutingProcess,
    SnowProcess,
    SoilProcess,
)
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


@dataclass(frozen=True)
class _HBVPreRouting:
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


class HBVModel(StateSpaceModel):
    """Single-zone, 14-parameter daily HBV-Light model."""

    n_zones = 1
    routing_buffer_size = constants.ROUTING_BUFFER_SIZE

    def __init__(
        self,
        *,
        tt: torch.Tensor | None = None,
        cfmax: torch.Tensor | None = None,
        sfcf: torch.Tensor | None = None,
        cwh: torch.Tensor | None = None,
        cfr: torch.Tensor | None = None,
        fc: torch.Tensor | None = None,
        lp: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        k0: torch.Tensor | None = None,
        k1: torch.Tensor | None = None,
        k2: torch.Tensor | None = None,
        perc: torch.Tensor | None = None,
        uzl: torch.Tensor | None = None,
        maxbas: torch.Tensor | None = None,
        snow: SnowProcess | None = None,
        soil: SoilProcess | None = None,
        response: ResponseProcess | None = None,
        routing: RoutingProcess | None = None,
        **replacement_parameters: torch.Tensor,
    ) -> None:
        super().__init__()
        self.snow = PhysicalSnowProcess() if snow is None else snow
        self.soil = PhysicalSoilProcess() if soil is None else soil
        self.response = PhysicalResponseProcess() if response is None else response
        self.routing = PhysicalRoutingProcess() if routing is None else routing

        self.parameter_bounds: dict[str, tuple[float, float]] = {}
        for slot in (self.snow, self.soil, self.response, self.routing):
            for name, bounds in slot.introduces.items():
                if name in self.parameter_bounds:
                    raise ValueError(f"parameter {name!r} is introduced by more than one installed HBV slot")
                self.parameter_bounds[name] = bounds

        parameter_values = {
            name: value
            for name, value in {
                "tt": tt,
                "cfmax": cfmax,
                "sfcf": sfcf,
                "cwh": cwh,
                "cfr": cfr,
                "fc": fc,
                "lp": lp,
                "beta": beta,
                "k0": k0,
                "k1": k1,
                "k2": k2,
                "perc": perc,
                "uzl": uzl,
                "maxbas": maxbas,
            }.items()
            if value is not None
        }
        parameter_values.update(replacement_parameters)
        missing = set(self.parameter_bounds) - set(parameter_values)
        unexpected = set(parameter_values) - set(self.parameter_bounds)
        if missing or unexpected:
            raise ValueError(
                f"parameter tensors must exactly match installed HBV slots; "
                f"missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r}"
            )
        for name in self.parameter_bounds:
            self.register_parameter(name, nn.Parameter(parameter_values[name]))

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

    def _pre_routing(
        self,
        state: HBVState,
        forcing: HBVForcing,
        parameters: Mapping[str, torch.Tensor],
    ) -> _HBVPreRouting:
        precip = forcing.precip
        pet = forcing.pet
        temp = forcing.temp

        p_rain, p_snow, new_sp, melt, new_lw, snow_input = self.snow(
            precip,
            temp,
            state.zone_sp,
            state.zone_lw,
            parameters,
        )
        new_sm, recharge_total, et_act = self.soil(
            snow_input,
            pet,
            state.zone_sm,
            parameters,
        )
        new_suz, new_slz, q0, q1, q2, perc, qgw = self.response(
            state.upper_zone,
            state.lower_zone,
            recharge_total,
            parameters,
        )
        return _HBVPreRouting(
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
        )

    def _apply_routing(
        self,
        qgw: torch.Tensor,
        routing_state: torch.Tensor,
        parameters: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.routing(routing_state, qgw, parameters)

    def transition(
        self,
        state: HBVState,
        forcing: HBVForcing,
        parameters: Mapping[str, torch.Tensor],
    ) -> tuple[HBVState, HBVFluxes]:
        pre_routing = self._pre_routing(state, forcing, parameters)
        qsim, new_buffer = self._apply_routing(
            pre_routing.qgw,
            state.routing_buffer,
            parameters,
        )

        new_state = HBVState(
            zone_sp=pre_routing.snow_pack,
            zone_lw=pre_routing.liquid_water_in_snow,
            zone_sm=pre_routing.soil_moisture,
            upper_zone=pre_routing.upper_zone,
            lower_zone=pre_routing.lower_zone,
            routing_buffer=new_buffer,
        )
        fluxes = HBVFluxes(
            precip=pre_routing.precip,
            temp=pre_routing.temp,
            pet=pre_routing.pet,
            precip_rain=pre_routing.precip_rain,
            precip_snow=pre_routing.precip_snow,
            snow_pack=pre_routing.snow_pack,
            snow_melt=pre_routing.snow_melt,
            liquid_water_in_snow=pre_routing.liquid_water_in_snow,
            snow_input=pre_routing.snow_input,
            soil_moisture=pre_routing.soil_moisture,
            recharge=pre_routing.recharge,
            actual_et=pre_routing.actual_et,
            upper_zone=pre_routing.upper_zone,
            lower_zone=pre_routing.lower_zone,
            q0=pre_routing.q0,
            q1=pre_routing.q1,
            q2=pre_routing.q2,
            percolation=pre_routing.percolation,
            qgw=pre_routing.qgw,
            streamflow=qsim,
        )
        return new_state, fluxes
