"""Small causal recurrent encoders. KDA adapted from FLA (MIT).

Source b8da4873fa0834849b64938af9f89bc35a347d90, fla/layers/kda.py and
fla/ops/kda/naive.py; see fla-LICENSE.txt. Uses the supported no-short-conv
configuration, one head, no accelerator-specific kernels.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional


def kda_recurrence(
    q: Tensor, k: Tensor, v: Tensor, log_decay: Tensor, beta: Tensor, initial: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Prepared inputs [batch,time,head,coordinate]; beta [batch,time,head]."""
    batch, _, heads, keys = q.shape
    state = q.new_zeros(batch, heads, keys, v.shape[-1]) if initial is None else initial
    outputs: list[Tensor] = []
    for qt, kt, vt, gt, bt in zip(
        q.unbind(1), k.unbind(1), v.unbind(1), log_decay.unbind(1), beta.unbind(1), strict=True
    ):
        state = state * gt.exp()[..., None]
        residual = vt - (kt[..., None] * state).sum(-2)
        state = state + bt[..., None, None] * kt[..., None] * residual[..., None, :]
        outputs.append((qt[..., None] * state).sum(-2) / math.sqrt(keys))
    return torch.stack(outputs, 1), state


class KDA(nn.Module):
    """Single-head Kimi Delta Attention with key-coordinate decay."""

    def __init__(self, inputs: int, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.q = nn.Linear(inputs, hidden, bias=False)
        self.k = nn.Linear(inputs, hidden, bias=False)
        self.v = nn.Linear(inputs, hidden, bias=False)
        self.decay = nn.Sequential(nn.Linear(inputs, hidden, bias=False), nn.Linear(hidden, hidden, bias=False))
        self.beta = nn.Linear(inputs, 1, bias=False)
        self.gate = nn.Sequential(nn.Linear(inputs, hidden, bias=False), nn.Linear(hidden, hidden))
        self.output = nn.Linear(hidden, hidden, bias=False)
        self.a_log = nn.Parameter(torch.empty(1).uniform_(1, 16).log())
        dt = torch.empty(hidden).uniform_(math.log(0.001), math.log(0.1)).exp()
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.norm_weight = nn.Parameter(torch.ones(hidden))

    def forward(  # ty: ignore[invalid-mutable-override] -- PyTorch declares forward as writable Callable.
        self, inputs: Tensor
    ) -> Tensor:
        q, k, v = (functional.silu(layer(inputs)) for layer in (self.q, self.k, self.v))
        q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        log_decay = -self.a_log.exp() * functional.softplus(self.decay(inputs) + self.dt_bias)
        output, _ = kda_recurrence(
            q.unsqueeze(2), k.unsqueeze(2), v.unsqueeze(2), log_decay.unsqueeze(2), self.beta(inputs).sigmoid()
        )
        output = output.squeeze(2)
        normalized = output * torch.rsqrt(output.square().mean(-1, keepdim=True) + 1e-5) * self.norm_weight
        return self.output(normalized * self.gate(inputs).sigmoid())


class LSTM(nn.Module):
    def __init__(self, inputs: int, hidden: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(inputs, hidden, batch_first=True)

    def forward(  # ty: ignore[invalid-mutable-override] -- PyTorch declares forward as writable Callable.
        self, inputs: Tensor
    ) -> Tensor:
        output, _ = self.lstm(inputs)
        return output
