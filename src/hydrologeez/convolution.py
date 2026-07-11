"""Shared PyTorch delay-line convolution for routing kernels."""

from __future__ import annotations

from typing import Any, overload

import torch


@overload
def convolve_delay_line(
    buffer: torch.Tensor,
    kernel: torch.Tensor,
    inflow: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def convolve_delay_line(buffer: Any, kernel: Any, inflow: Any) -> Any: ...


def convolve_delay_line(
    buffer: torch.Tensor,
    kernel: torch.Tensor,
    inflow: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift, inject the current inflow, then read the delay-line head."""
    shifted = torch.cat((buffer[..., 1:], torch.zeros_like(buffer[..., :1])), dim=-1)
    new_buffer = shifted + kernel * inflow.unsqueeze(-1)
    output = new_buffer[..., 0]
    return output, new_buffer
