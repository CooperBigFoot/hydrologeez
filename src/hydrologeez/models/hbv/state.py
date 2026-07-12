"""HBV-Light single-zone state carrier."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .constants import ROUTING_BUFFER_SIZE


@dataclass(frozen=True)
class HBVState:
    """HBV state using the Rust-compatible flat state layout."""

    zone_sp: torch.Tensor
    zone_lw: torch.Tensor
    zone_sm: torch.Tensor
    upper_zone: torch.Tensor
    lower_zone: torch.Tensor
    routing_buffer: torch.Tensor

    def to_flat(self) -> torch.Tensor:
        """Pack as ``[..., SP, LW, SM, SUZ, SLZ, b0..b6]``."""
        return torch.cat(
            (
                self.zone_sp.unsqueeze(-1),
                self.zone_lw.unsqueeze(-1),
                self.zone_sm.unsqueeze(-1),
                self.upper_zone.unsqueeze(-1),
                self.lower_zone.unsqueeze(-1),
                self.routing_buffer,
            ),
            dim=-1,
        )

    @classmethod
    def from_flat(cls, flat: torch.Tensor) -> HBVState:
        """Reconstruct from the Rust-compatible layout along the last axis."""
        return cls(
            zone_sp=flat[..., 0],
            zone_lw=flat[..., 1],
            zone_sm=flat[..., 2],
            upper_zone=flat[..., 3],
            lower_zone=flat[..., 4],
            routing_buffer=flat[..., 5 : 5 + ROUTING_BUFFER_SIZE],
        )
