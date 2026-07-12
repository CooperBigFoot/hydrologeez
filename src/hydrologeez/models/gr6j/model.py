"""GR6J as a differentiable state-space model on the hydrologeez SSM base."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from hydrologeez.models.gr6j import constants, processes
from hydrologeez.models.gr6j.state import State
from hydrologeez.ssm import StateSpaceModel

B = constants.B
C = constants.C


@dataclass(frozen=True)
class GR6JForcing:
    """Per-step or per-series GR6J forcing."""

    precip: torch.Tensor
    pet: torch.Tensor


@dataclass(frozen=True)
class GR6JFluxes:
    """All GR6J internal fluxes for one step, stacked over time by the SSM."""

    pet: torch.Tensor
    precip: torch.Tensor
    production_store: torch.Tensor
    net_rainfall: torch.Tensor
    storage_infiltration: torch.Tensor
    actual_et: torch.Tensor
    percolation: torch.Tensor
    effective_rainfall: torch.Tensor
    q9: torch.Tensor
    q1: torch.Tensor
    routing_store: torch.Tensor
    exchange: torch.Tensor
    actual_exchange_routing: torch.Tensor
    actual_exchange_direct: torch.Tensor
    actual_exchange_total: torch.Tensor
    qr: torch.Tensor
    qrexp: torch.Tensor
    exponential_store: torch.Tensor
    qd: torch.Tensor
    streamflow: torch.Tensor


class GR6J(StateSpaceModel):
    """GR6J six-parameter daily rainfall-runoff model as a Torch module."""

    def __init__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
        x6: torch.Tensor,
        nh: int = constants.NH,
    ) -> None:
        super().__init__()
        self.x1 = nn.Parameter(x1)
        self.x2 = nn.Parameter(x2)
        self.x3 = nn.Parameter(x3)
        self.x4 = nn.Parameter(x4)
        self.x5 = nn.Parameter(x5)
        self.x6 = nn.Parameter(x6)
        self.nh = nh

    def init_state(
        self,
        parameters: Mapping[str, torch.Tensor],
        *,
        batch_size: int,
    ) -> State:
        x1 = parameters["x1"]
        x3 = parameters["x3"]
        if x1.ndim == 2:
            x1 = x1[:, 0]
        if x3.ndim == 2:
            x3 = x3[:, 0]
        if x1.ndim == 0:
            x1 = x1.expand(batch_size)
        if x3.ndim == 0:
            x3 = x3.expand(batch_size)
        return State(
            production_store=0.3 * x1,
            routing_store=0.5 * x3,
            exponential_store=x1.new_zeros(batch_size),
            uh1=x1.new_zeros((batch_size, self.nh)),
            uh2=x1.new_zeros((batch_size, 2 * self.nh)),
        )

    def transition(
        self,
        state: State,
        forcing: GR6JForcing,
        parameters: Mapping[str, torch.Tensor],
    ) -> tuple[State, GR6JFluxes]:
        x1, x2, x3, x4, x5, x6 = (parameters[f"x{index}"] for index in range(1, 7))
        precip = forcing.precip
        pet = forcing.pet

        uh1_ord, uh2_ord = processes.compute_uh_ordinates(x4)

        s_after_ps, actual_et, net_rainfall_pn, effective_rainfall_pr = processes.production_store_update(
            precip, pet, state.production_store, x1
        )
        storage_infiltration = net_rainfall_pn - effective_rainfall_pr

        s_after_perc, percolation_amount = processes.percolation(s_after_ps, x1)
        total_effective_rainfall = effective_rainfall_pr + percolation_amount

        q9, uh1 = processes.convolve_uh(state.uh1, uh1_ord, B * total_effective_rainfall)
        q1, uh2 = processes.convolve_uh(state.uh2, uh2_ord, (1.0 - B) * total_effective_rainfall)

        exchange_f = processes.groundwater_exchange(state.routing_store, x2, x3, x5)

        new_routing_store, qr, actual_exchange_routing = processes.routing_store_update(
            state.routing_store, (1.0 - C) * q9, exchange_f, x3
        )
        new_exp_store, qrexp = processes.exponential_store_update(state.exponential_store, C * q9, exchange_f, x6)
        qd, actual_exchange_direct = processes.direct_branch(q1, exchange_f)

        streamflow = torch.clamp_min(qr + qrexp + qd, 0.0)
        actual_exchange_total = actual_exchange_routing + actual_exchange_direct + exchange_f

        new_state = State(
            production_store=s_after_perc,
            routing_store=new_routing_store,
            exponential_store=new_exp_store,
            uh1=uh1,
            uh2=uh2,
        )
        fluxes = GR6JFluxes(
            pet=pet,
            precip=precip,
            production_store=s_after_perc,
            net_rainfall=net_rainfall_pn,
            storage_infiltration=storage_infiltration,
            actual_et=actual_et,
            percolation=percolation_amount,
            effective_rainfall=total_effective_rainfall,
            q9=q9,
            q1=q1,
            routing_store=new_routing_store,
            exchange=exchange_f,
            actual_exchange_routing=actual_exchange_routing,
            actual_exchange_direct=actual_exchange_direct,
            actual_exchange_total=actual_exchange_total,
            qr=qr,
            qrexp=qrexp,
            exponential_store=new_exp_store,
            qd=qd,
            streamflow=streamflow,
        )
        return new_state, fluxes
