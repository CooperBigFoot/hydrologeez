"""Single-zone HBV-Light as a differentiable state-space model on the SSM base.

Pure JAX/Equinox implementation. Lumped (n_zones=1); elevation extrapolation
is bypassed (input_elevation=None -> zone_temp=temp, zone_precip=precip).
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import torch

from hydrologeez.models.hbv import constants, processes
from hydrologeez.models.hbv.state import HBVState
from hydrologeez.ssm import StateSpaceModel


@dataclass(frozen=True)
class HBVForcing:
    """Per-step or per-series HBV forcing (3 leaves)."""

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
    """Single-zone (lumped) HBV-Light, 14-parameter daily model, as an Equinox module.

    Parameters are the 14 fields tt..maxbas (canonical order). ``n_zones`` and
    ``routing_buffer_size`` are STATIC structural integers (never calibrated).
    """

    tt: jax.Array
    cfmax: jax.Array
    sfcf: jax.Array
    cwh: jax.Array
    cfr: jax.Array
    fc: jax.Array
    lp: jax.Array
    beta: jax.Array
    k0: jax.Array
    k1: jax.Array
    k2: jax.Array
    perc: jax.Array
    uzl: jax.Array
    maxbas: jax.Array
    n_zones: int = eqx.field(static=True, default=1)
    routing_buffer_size: int = eqx.field(static=True, default=constants.ROUTING_BUFFER_SIZE)

    def init_state(self) -> HBVState:
        """Rust State::initialize (state.rs:28-40): SM=0.5*fc, all else zero."""
        return HBVState(
            zone_sp=jnp.asarray(0.0),
            zone_lw=jnp.asarray(0.0),
            zone_sm=0.5 * self.fc,
            upper_zone=jnp.asarray(0.0),
            lower_zone=jnp.asarray(0.0),
            routing_buffer=jnp.zeros(self.routing_buffer_size),
        )

    def transition(self, state: HBVState, forcing: HBVForcing) -> tuple[HBVState, HBVFluxes]:
        precip = forcing.precip
        pet = forcing.pet
        temp = forcing.temp

        uh_weights = processes.compute_triangular_weights(self.maxbas)

        # --- Snow routine (reads start-of-step sp, lw) --- run.rs:195-203
        p_rain, p_snow = processes.partition_precipitation(precip, temp, self.tt, self.sfcf)  # ty: ignore[invalid-argument-type]
        melt = processes.compute_melt(temp, self.tt, self.cfmax, state.zone_sp)  # ty: ignore[invalid-argument-type]
        refreeze = processes.compute_refreezing(temp, self.tt, self.cfmax, self.cfr, state.zone_lw)  # ty: ignore[invalid-argument-type]
        new_sp, new_lw, snow_outflow = processes.update_snow_pack(
            state.zone_sp, state.zone_lw, p_snow, melt, refreeze, self.cwh
        )
        snow_input = p_rain + snow_outflow

        # --- Soil routine (recharge + ET + update all read start-of-step sm) --- Seibert & Vis (2012)
        recharge = processes.compute_recharge(snow_input, state.zone_sm, self.fc, self.beta)
        et_act = processes.compute_actual_et(pet, state.zone_sm, self.fc, self.lp)  # ty: ignore[invalid-argument-type]
        new_sm, sm_overflow = processes.update_soil_moisture(state.zone_sm, snow_input, recharge, et_act, self.fc)
        # Above-FC excess routes to upper-zone recharge (not discarded). The reported
        # recharge flux is the TOTAL soil->upper-zone flux (base recharge + overflow).
        recharge_total = recharge + sm_overflow

        # --- Response routine (all read OLD SUZ/SLZ; explicit operator-splitting) --- run.rs:214-222
        q0, q1 = processes.upper_zone_outflows(state.upper_zone, self.k0, self.k1, self.uzl)
        perc = processes.compute_percolation(state.upper_zone, self.perc)
        new_suz = processes.update_upper_zone(state.upper_zone, recharge_total, q0, q1, perc)
        q2 = processes.lower_zone_outflow(state.lower_zone, self.k2)
        new_slz = processes.update_lower_zone(state.lower_zone, perc, q2)
        qgw = q0 + q1 + q2

        # --- Routing (read-after-shift; same-day ordinate-1 term, no forced lag) ---
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
            precip_rain=p_rain,  # ty: ignore[invalid-argument-type]
            precip_snow=p_snow,  # ty: ignore[invalid-argument-type]
            snow_pack=new_sp,  # ty: ignore[invalid-argument-type]
            snow_melt=melt,  # ty: ignore[invalid-argument-type]
            liquid_water_in_snow=new_lw,  # ty: ignore[invalid-argument-type]
            snow_input=snow_input,  # ty: ignore[invalid-argument-type]
            soil_moisture=new_sm,  # ty: ignore[invalid-argument-type]
            recharge=recharge_total,  # ty: ignore[invalid-argument-type]
            actual_et=et_act,  # ty: ignore[invalid-argument-type]
            upper_zone=new_suz,  # ty: ignore[invalid-argument-type]
            lower_zone=new_slz,  # ty: ignore[invalid-argument-type]
            q0=q0,  # ty: ignore[invalid-argument-type]
            q1=q1,  # ty: ignore[invalid-argument-type]
            q2=q2,  # ty: ignore[invalid-argument-type]
            percolation=perc,  # ty: ignore[invalid-argument-type]
            qgw=qgw,  # ty: ignore[invalid-argument-type]
            streamflow=qsim,  # ty: ignore[invalid-argument-type]
        )
        return new_state, fluxes
