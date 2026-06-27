"""Abstract state-space-model base for hydrologeez.

Discrete-time state-space form (DESIGN.md):

    transition:  (state, forcing) -> (state, fluxes)   # fused; ALL internal fluxes
    observation: (state, fluxes)  -> observable        # pluggable; default streamflow
    run:         lax.scan(transition) over forcing     # the fold

Subclasses implement exactly two methods: ``init_state`` and ``transition``.
``run`` (lax.scan) and ``batch_run`` (vmap) are provided here for free.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
from jax import lax

from hydrologeez.observation import default_streamflow_observation

# Documentation-level aliases: these are model-defined PyTrees.
State = Any
Forcing = Any
Fluxes = Any
Observable = Any
ObservationOperator = Callable[[State, Fluxes], Observable]


class StateSpaceModel(eqx.Module):
    """Abstract discrete-time state-space hydrological model.

    A model IS an ``eqx.Module`` whose fields are its parameters, so
    ``jax.grad(loss)(model)`` differentiates through them.

    Contributor contract -- implement ONLY:

    * ``init_state(self) -> State``: the initial carry for ``lax.scan``.
    * ``transition(self, state, forcing) -> (state, fluxes)``: one fused step
      returning the next state and ALL internal fluxes.

    Static-shape policy (hard JAX rule): any structural integer that controls an
    array shape (e.g. unit-hydrograph length / number of elevation bands) MUST be
    declared ``eqx.field(static=True)`` and set at construction -- it is NEVER
    calibrated. Continuous shape-affecting parameters (e.g. GR6J ``x4``) instead
    use fixed-length MASKED kernels sized to a declared upper bound, with
    ordinates computed as a smooth function of the parameter and zero-padded
    beyond active support (see DESIGN.md). Process math lives in plain free
    functions in a per-model ``processes.py``, not here.
    """

    @abc.abstractmethod
    def init_state(self) -> State:
        raise NotImplementedError

    @abc.abstractmethod
    def transition(self, state: State, forcing: Forcing) -> tuple[State, Fluxes]:
        raise NotImplementedError

    def run(
        self,
        forcing: Forcing,
        *,
        observation_operator: ObservationOperator = default_streamflow_observation,
        return_fluxes: bool = False,
    ):
        """Fold ``transition`` over ``forcing`` with ``lax.scan``.

        ``forcing`` is a PyTree whose leaves carry a leading time axis of length
        T. Returns the observable timeseries (leading axis T). With
        ``return_fluxes=True`` returns ``(observable, fluxes, final_state)`` where
        ``fluxes`` is the stacked per-step fluxes PyTree.
        """
        init_state = self.init_state()

        def step(carry: State, u: Forcing):
            new_state, fluxes = self.transition(carry, u)
            observable = observation_operator(new_state, fluxes)
            return new_state, (observable, fluxes)

        final_state, (observable, fluxes) = lax.scan(step, init_state, forcing)
        if return_fluxes:
            return observable, fluxes, final_state
        return observable

    def batch_run(
        self,
        forcings: Forcing,
        *,
        observation_operator: ObservationOperator = default_streamflow_observation,
        return_fluxes: bool = False,
    ):
        """Vectorise ``run`` over a leading batch axis of ``forcings`` via vmap.

        The model itself is held constant (closed over); only ``forcings`` is
        mapped. Equivalent to looping ``run`` over each batch member, but compiled
        into one batched call.
        """

        def _one(forcing: Forcing):
            return self.run(
                forcing,
                observation_operator=observation_operator,
                return_fluxes=return_fluxes,
            )

        return jax.vmap(_one)(forcings)
