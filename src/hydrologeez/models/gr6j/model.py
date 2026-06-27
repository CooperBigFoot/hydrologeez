"""GR6J as a differentiable state-space model on the hydrologeez SSM base."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from hydrologeez.models.gr6j import constants, processes
from hydrologeez.models.gr6j.state import State
from hydrologeez.ssm import StateSpaceModel

B = constants.B
C = constants.C


class GR6JForcing(eqx.Module):
    """Per-step or per-series GR6J forcing."""

    precip: jax.Array
    pet: jax.Array


class GR6JFluxes(eqx.Module):
    """All GR6J internal fluxes for one step, stacked over time by scan."""

    pet: jax.Array
    precip: jax.Array
    production_store: jax.Array
    net_rainfall: jax.Array
    storage_infiltration: jax.Array
    actual_et: jax.Array
    percolation: jax.Array
    effective_rainfall: jax.Array
    q9: jax.Array
    q1: jax.Array
    routing_store: jax.Array
    exchange: jax.Array
    actual_exchange_routing: jax.Array
    actual_exchange_direct: jax.Array
    actual_exchange_total: jax.Array
    qr: jax.Array
    qrexp: jax.Array
    exponential_store: jax.Array
    qd: jax.Array
    streamflow: jax.Array


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
            precip, pet, state.production_store, self.x1
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
        actual_exchange_total = actual_exchange_routing + actual_exchange_direct

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
