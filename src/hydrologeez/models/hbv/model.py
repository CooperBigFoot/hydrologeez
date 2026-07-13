"""Single-zone HBV-Light as a differentiable PyTorch state-space model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

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


def _component_forcing_shape(forcing: HBVForcing, *, allow_empty: bool) -> tuple[int, int]:
    leaves = (forcing.precip, forcing.pet, forcing.temp)
    if any(leaf.ndim != 2 for leaf in leaves):
        raise ValueError("HBV forcing tensor leaves must have shape [B, T]")
    shape = leaves[0].shape
    if any(leaf.shape != shape for leaf in leaves[1:]):
        raise ValueError("HBV forcing tensor leaves must share shape [B, T]")
    if not allow_empty and shape[1] == 0:
        raise ValueError("main forcing must contain at least one timestep")
    return cast(tuple[int, int], shape)


def _expand_component_parameters(
    parameters: Mapping[str, torch.Tensor],
    parameter_bounds: Mapping[str, tuple[float, float]],
    *,
    batch_size: int,
    n_components: int,
    time_steps: int,
) -> dict[str, torch.Tensor]:
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    missing = set(parameter_bounds) - set(parameters)
    unexpected = set(parameters) - set(parameter_bounds)
    if missing or unexpected:
        raise ValueError(
            "component parameters must exactly match installed HBV slots; "
            f"missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r}"
        )

    expanded: dict[str, torch.Tensor] = {}
    for name in parameter_bounds:
        value = parameters[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"parameter {name!r} must be a torch.Tensor")
        if value.ndim == 0:
            expanded[name] = value.expand(batch_size, n_components, time_steps)
        elif value.ndim == 1:
            if value.shape != (batch_size,):
                raise ValueError(f"parameter {name!r} batch size does not match forcing")
            expanded[name] = value[:, None, None].expand(batch_size, n_components, time_steps)
        elif value.ndim == 2:
            if value.shape != (batch_size, time_steps):
                raise ValueError(f"parameter {name!r} must have shape [B, T]")
            expanded[name] = value[:, None, :].expand(batch_size, n_components, time_steps)
        elif value.ndim == 3:
            if value.shape != (batch_size, n_components, time_steps):
                raise ValueError(f"parameter {name!r} must have shape [B, N, T]")
            expanded[name] = value
        else:
            raise ValueError(f"parameter {name!r} must have shape [], [B], [B, T], or [B, N, T]")
    return expanded


def _component_parameters_at(parameters: Mapping[str, torch.Tensor], time: int) -> dict[str, torch.Tensor]:
    return {name: value[:, :, time] for name, value in parameters.items()}


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

    def _init_component_state(
        self,
        parameters: Mapping[str, torch.Tensor],
        *,
        batch_size: int,
        n_components: int,
    ) -> HBVState:
        flat_parameters = {name: value.reshape(batch_size * n_components) for name, value in parameters.items()}
        flat_state = self.init_state(flat_parameters, batch_size=batch_size * n_components)
        return HBVState(
            zone_sp=flat_state.zone_sp.reshape(batch_size, n_components),
            zone_lw=flat_state.zone_lw.reshape(batch_size, n_components),
            zone_sm=flat_state.zone_sm.reshape(batch_size, n_components),
            upper_zone=flat_state.upper_zone.reshape(batch_size, n_components),
            lower_zone=flat_state.lower_zone.reshape(batch_size, n_components),
            routing_buffer=flat_state.routing_buffer.reshape(batch_size, n_components, self.routing_buffer_size)[
                :, 0, :
            ],
        )

    def _component_step(
        self,
        state: HBVState,
        forcing: HBVForcing,
        parameters: Mapping[str, torch.Tensor],
        *,
        batch_size: int,
        n_components: int,
    ) -> tuple[HBVState, torch.Tensor]:
        flat_batch_size = batch_size * n_components
        flat_state = HBVState(
            zone_sp=state.zone_sp.reshape(flat_batch_size),
            zone_lw=state.zone_lw.reshape(flat_batch_size),
            zone_sm=state.zone_sm.reshape(flat_batch_size),
            upper_zone=state.upper_zone.reshape(flat_batch_size),
            lower_zone=state.lower_zone.reshape(flat_batch_size),
            routing_buffer=state.routing_buffer,
        )
        flat_forcing = HBVForcing(
            precip=forcing.precip[:, None].expand(batch_size, n_components).reshape(flat_batch_size),
            pet=forcing.pet[:, None].expand(batch_size, n_components).reshape(flat_batch_size),
            temp=forcing.temp[:, None].expand(batch_size, n_components).reshape(flat_batch_size),
        )
        flat_parameters = {name: value.reshape(flat_batch_size) for name, value in parameters.items()}
        pre_routing = self._pre_routing(flat_state, flat_forcing, flat_parameters)

        mean_qgw = pre_routing.qgw.reshape(batch_size, n_components).mean(dim=1)
        routing_parameters = {name: parameters[name].mean(dim=1) for name in self.routing.introduces}
        streamflow, routing_buffer = self._apply_routing(
            mean_qgw,
            state.routing_buffer,
            routing_parameters,
        )
        return (
            HBVState(
                zone_sp=pre_routing.snow_pack.reshape(batch_size, n_components),
                zone_lw=pre_routing.liquid_water_in_snow.reshape(batch_size, n_components),
                zone_sm=pre_routing.soil_moisture.reshape(batch_size, n_components),
                upper_zone=pre_routing.upper_zone.reshape(batch_size, n_components),
                lower_zone=pre_routing.lower_zone.reshape(batch_size, n_components),
                routing_buffer=routing_buffer,
            ),
            streamflow,
        )

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

    def run_components(
        self,
        forcing: HBVForcing,
        parameters: Mapping[str, torch.Tensor],
        n_components: int,
        *,
        warmup: HBVForcing | None = None,
        warmup_parameters: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run HBV components with average-before-routing shared routing.

        Component parameters may have shape ``[]``, ``[B]``, ``[B,T]``, or
        ``[B,N,T]``. Snow, soil, and response stores have shape ``[B,N]``;
        their groundwater runoff is averaged over ``N`` before routing. Routing
        is applied exactly once per basin with the component-mean routing
        parameters and one shared routing buffer of shape ``[B,7]``, never one
        routing buffer per component.
        """
        batch_size, time_steps = _component_forcing_shape(forcing, allow_empty=False)
        resolved = _expand_component_parameters(
            parameters,
            self.parameter_bounds,
            batch_size=batch_size,
            n_components=n_components,
            time_steps=time_steps,
        )

        if warmup is None:
            state = self._init_component_state(
                _component_parameters_at(resolved, 0),
                batch_size=batch_size,
                n_components=n_components,
            )
        else:
            warmup_batch, warmup_steps = _component_forcing_shape(warmup, allow_empty=False)
            if warmup_batch != batch_size:
                raise ValueError("warmup batch size must match main forcing")
            resolved_warmup = _expand_component_parameters(
                parameters if warmup_parameters is None else warmup_parameters,
                self.parameter_bounds,
                batch_size=batch_size,
                n_components=n_components,
                time_steps=warmup_steps,
            )
            state = self._init_component_state(
                _component_parameters_at(resolved_warmup, 0),
                batch_size=batch_size,
                n_components=n_components,
            )
            with torch.no_grad():
                for time in range(warmup_steps):
                    state, _ = self._component_step(
                        state,
                        HBVForcing(
                            precip=warmup.precip[:, time],
                            pet=warmup.pet[:, time],
                            temp=warmup.temp[:, time],
                        ),
                        _component_parameters_at(resolved_warmup, time),
                        batch_size=batch_size,
                        n_components=n_components,
                    )
            state = HBVState(
                zone_sp=state.zone_sp.detach(),
                zone_lw=state.zone_lw.detach(),
                zone_sm=state.zone_sm.detach(),
                upper_zone=state.upper_zone.detach(),
                lower_zone=state.lower_zone.detach(),
                routing_buffer=state.routing_buffer.detach(),
            )

        observations: list[torch.Tensor] = []
        for time in range(time_steps):
            state, streamflow = self._component_step(
                state,
                HBVForcing(
                    precip=forcing.precip[:, time],
                    pet=forcing.pet[:, time],
                    temp=forcing.temp[:, time],
                ),
                _component_parameters_at(resolved, time),
                batch_size=batch_size,
                n_components=n_components,
            )
            observations.append(streamflow)
        return torch.stack(observations, dim=1)
