"""Pluggable observation operators for hydrologeez state-space models.

An observation operator maps ``(state, fluxes) -> observable``. The canonical
default returns streamflow. Models reference ``default_streamflow_observation``
rather than redefining the default, so there is a SINGLE source of truth for
the default observable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

# Documentation-level aliases: state and fluxes are model-defined tensor containers.
State = Any
Fluxes = Any
Observable = torch.Tensor
ObservationOperator = Callable[[State, Fluxes], torch.Tensor]


def default_streamflow_observation(state: State, fluxes: Fluxes) -> Observable:
    """Canonical default observation operator: return the ``streamflow`` flux.

    The ``fluxes`` container must expose a ``streamflow`` attribute. The GR6J fluxes
    type (added in a later step) and the linear-reservoir test dummy both satisfy
    this contract.
    """
    return fluxes.streamflow
