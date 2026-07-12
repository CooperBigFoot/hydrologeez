"""Explicit tensor dtype and device policies.

The reference path is float64 on CPU. The training path is float32 on an
explicitly selected device. Helpers return or create local values only and never
mutate torch's process-wide defaults.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import Tensor

REFERENCE_DTYPE = torch.float64
REFERENCE_DEVICE = torch.device("cpu")
TRAINING_DTYPE = torch.float32

ENABLE_FIX_MESSAGE = (
    "Import-time x64 enforcement has been retired; use reference_tensor() for the "
    "float64 CPU reference path or training_tensor() for float32 training."
)

__all__ = [
    "ENABLE_FIX_MESSAGE",
    "REFERENCE_DEVICE",
    "REFERENCE_DTYPE",
    "TRAINING_DTYPE",
    "enforce_float64",
    "reference_defaults",
    "reference_tensor",
    "training_defaults",
    "training_tensor",
]


def reference_defaults() -> dict[str, torch.dtype | torch.device]:
    """Return explicit kwargs for float64 CPU reference computations."""
    return {"dtype": REFERENCE_DTYPE, "device": REFERENCE_DEVICE}


def training_defaults(device: torch.device | str) -> dict[str, torch.dtype | torch.device]:
    """Return explicit kwargs for float32 training on ``device``."""
    return {"dtype": TRAINING_DTYPE, "device": torch.device(device)}


def reference_tensor(data: Any) -> Tensor:
    """Convert ``data`` to a float64 CPU tensor without changing global defaults."""
    return torch.as_tensor(data).to(dtype=REFERENCE_DTYPE, device=REFERENCE_DEVICE)


def training_tensor(data: Any, *, device: torch.device | str) -> Tensor:
    """Convert ``data`` to a float32 tensor on ``device``."""
    return torch.as_tensor(data).to(dtype=TRAINING_DTYPE, device=torch.device(device))


def enforce_float64() -> None:
    """Deprecated compatibility no-op for the retired import-time float64 guard."""
    warnings.warn(ENABLE_FIX_MESSAGE, DeprecationWarning, stacklevel=2)
