"""GR6J Torch state container."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .constants import STATE_SIZE, UH1_LEN, UH2_LEN


@dataclass(frozen=True)
class State:
    """GR6J state tensors with optional shared leading batch dimensions."""

    production_store: torch.Tensor
    routing_store: torch.Tensor
    exponential_store: torch.Tensor
    uh1: torch.Tensor
    uh2: torch.Tensor

    def to_flat(self) -> torch.Tensor:
        """Pack the last dimension as ``[S, R, Exp, uh1, uh2]``."""
        return torch.cat(
            (
                self.production_store.unsqueeze(-1),
                self.routing_store.unsqueeze(-1),
                self.exponential_store.unsqueeze(-1),
                self.uh1,
                self.uh2,
            ),
            dim=-1,
        )

    @classmethod
    def from_flat(cls, arr: torch.Tensor) -> State:
        """Reconstruct a state from the Rust 63-element last dimension."""
        if arr.shape[-1] != STATE_SIZE:
            raise ValueError(f"Expected last dimension of size {STATE_SIZE}, got {arr.shape[-1]}")
        return cls(
            production_store=arr[..., 0],
            routing_store=arr[..., 1],
            exponential_store=arr[..., 2],
            uh1=arr[..., 3 : 3 + UH1_LEN],
            uh2=arr[..., 3 + UH1_LEN : 3 + UH1_LEN + UH2_LEN],
        )
