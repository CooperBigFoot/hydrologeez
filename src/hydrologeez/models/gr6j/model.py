"""GR6J as a differentiable state-space model on the hydrologeez SSM base."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import torch

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
    """GR6J six-parameter daily rainfall-runoff model as an Equinox module."""

    x1: jax.Array
    x2: jax.Array
    x3: jax.Array
    x4: jax.Array
    x5: jax.Array
    x6: jax.Array
    nh: int = eqx.field(static=True, default=constants.NH)

    def init_state(self) -> State:
        return State(
            production_store=0.3 * self.x1,
            routing_store=0.5 * self.x3,
            exponential_store=jnp.asarray(0.0),
            uh1=jnp.zeros(self.nh),
            uh2=jnp.zeros(2 * self.nh),
        )

    def transition(self, state: State, forcing: GR6JForcing) -> tuple[State, GR6JFluxes]:
        precip = forcing.precip
        pet = forcing.pet

        uh1_ord, uh2_ord = processes.compute_uh_ordinates(self.x4)

        s_after_ps, actual_et, net_rainfall_pn, effective_rainfall_pr = processes.production_store_update(
            precip,  # ty: ignore[invalid-argument-type]
            pet,  # ty: ignore[invalid-argument-type]
            state.production_store,
            self.x1,
        )
        storage_infiltration = net_rainfall_pn - effective_rainfall_pr

        s_after_perc, percolation_amount = processes.percolation(s_after_ps, self.x1)
        total_effective_rainfall = effective_rainfall_pr + percolation_amount

        q9, uh1 = processes.convolve_uh(state.uh1, uh1_ord, B * total_effective_rainfall)
        q1, uh2 = processes.convolve_uh(state.uh2, uh2_ord, (1.0 - B) * total_effective_rainfall)

        exchange_f = processes.groundwater_exchange(state.routing_store, self.x2, self.x3, self.x5)

        new_routing_store, qr, actual_exchange_routing = processes.routing_store_update(
            state.routing_store, (1.0 - C) * q9, exchange_f, self.x3
        )
        new_exp_store, qrexp = processes.exponential_store_update(state.exponential_store, C * q9, exchange_f, self.x6)
        qd, actual_exchange_direct = processes.direct_branch(q1, exchange_f)

        streamflow = jnp.maximum(qr + qrexp + qd, 0.0)
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
            production_store=s_after_perc,  # ty: ignore[invalid-argument-type]
            net_rainfall=net_rainfall_pn,  # ty: ignore[invalid-argument-type]
            storage_infiltration=storage_infiltration,  # ty: ignore[invalid-argument-type]
            actual_et=actual_et,  # ty: ignore[invalid-argument-type]
            percolation=percolation_amount,  # ty: ignore[invalid-argument-type]
            effective_rainfall=total_effective_rainfall,  # ty: ignore[invalid-argument-type]
            q9=q9,  # ty: ignore[invalid-argument-type]
            q1=q1,  # ty: ignore[invalid-argument-type]
            routing_store=new_routing_store,  # ty: ignore[invalid-argument-type]
            exchange=exchange_f,  # ty: ignore[invalid-argument-type]
            actual_exchange_routing=actual_exchange_routing,  # ty: ignore[invalid-argument-type]
            actual_exchange_direct=actual_exchange_direct,  # ty: ignore[invalid-argument-type]
            actual_exchange_total=actual_exchange_total,  # ty: ignore[invalid-argument-type]
            qr=qr,  # ty: ignore[invalid-argument-type]
            qrexp=qrexp,  # ty: ignore[invalid-argument-type]
            exponential_store=new_exp_store,  # ty: ignore[invalid-argument-type]
            qd=qd,  # ty: ignore[invalid-argument-type]
            streamflow=streamflow,  # ty: ignore[invalid-argument-type]
        )
        return new_state, fluxes
