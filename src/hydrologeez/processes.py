"""Shared contracts for replaceable hydrological processes."""

from __future__ import annotations

import math
from abc import ABC
from typing import Any, ClassVar

from torch import nn

ParameterBounds = tuple[float, float]


class Process(nn.Module, ABC):
    """Abstract base for a process slot and its introduced parameters."""

    introduces: ClassVar[dict[str, ParameterBounds]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not isinstance(cls.introduces, dict):
            raise TypeError("introduces must be an insertion-ordered dict")
        for name, bounds in cls.introduces.items():
            if not isinstance(name, str) or not name:
                raise ValueError("introduced parameter names must be non-empty strings")
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError(f"bounds for {name!r} must be a (low, high) tuple")
            low, high = bounds
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in bounds
            ):
                raise ValueError(f"bounds for {name!r} must be finite numbers")
            if low >= high:
                raise ValueError(f"bounds for {name!r} must satisfy low < high")
