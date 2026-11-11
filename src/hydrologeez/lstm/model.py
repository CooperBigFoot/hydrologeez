"""Single-layer causal LSTM mapping daily meteorology and statics to discharge."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from hydrologeez.hybrid.data import Normalization

OutputActivation = Literal["softplus", "linear"]


@dataclass(frozen=True)
class ModelConfig:
    features: tuple[str, ...] = ("P", "T", "PET")
    attributes: tuple[str, ...] = ()
    hidden: int = 16
    output_activation: OutputActivation = "softplus"


class DischargeLSTM(nn.Module):
    """Direct daily discharge prediction [day, basin] without a physical bucket model.

    Non-padded meteorological values must be finite; unlike the hybrid pretext
    encoder, this network receives direct normalized meteorological channels
    plus caller-scaled static attributes, without missing/hidden indicator
    channels. ``output_activation='softplus'`` enforces strictly positive
    smooth discharge without silent post-hoc clipping; ``'linear'`` exposes raw
    signed linear outputs.
    """

    mean: Tensor
    scale: Tensor

    def __init__(self, config: ModelConfig, normalization: Normalization) -> None:
        super().__init__()
        if config.hidden < 1 or config.output_activation not in ("softplus", "linear"):
            raise ValueError("invalid hidden size or output activation")
        if (
            not config.features
            or config.features != normalization.features
            or len(set(config.attributes)) != len(config.attributes)
        ):
            raise ValueError("feature/attribute order must be explicit and unique")
        self.config = config
        self.register_buffer("mean", normalization.mean.detach().clone())
        self.register_buffer("scale", normalization.scale.detach().clone())
        inputs = len(config.features) + len(config.attributes)
        self.lstm = nn.LSTM(inputs, config.hidden, batch_first=True)
        self.readout = nn.Linear(config.hidden, 1)

    @property
    def normalization(self) -> Normalization:
        return Normalization(self.mean, self.scale, self.config.features)

    def forward(  # ty: ignore[invalid-mutable-override] -- PyTorch declares forward as writable Callable.
        self, values: Tensor, attributes: Tensor, padding: Tensor
    ) -> Tensor:
        if (
            values.ndim != 3
            or values.shape[0] == 0
            or values.shape[1] == 0
            or values.shape[2] != len(self.config.features)
        ):
            raise ValueError("meteorology must be nonempty [basin, day, feature] in configured order")
        if attributes.shape != (values.shape[0], len(self.config.attributes)) or not bool(
            torch.isfinite(attributes).all()
        ):
            raise ValueError("attributes must be finite [basin, attribute] in configured order")
        if (
            padding.shape != values.shape[:2]
            or padding.dtype != torch.bool
            or bool(padding[:, 0].any())
            or bool((padding[:, :-1] & ~padding[:, 1:]).any())
        ):
            raise ValueError("only right padding of nonempty sequences is supported")
        active = ~padding[..., None]
        if not bool(torch.isfinite(values[active.expand_as(values)]).all()):
            raise ValueError("active meteorological inputs must be finite; no missing-indicator channel is present")
        normalized = torch.where(active, (values - self.mean) / self.scale, 0.0)
        inputs = torch.cat((normalized, attributes[:, None, :].expand(-1, values.shape[1], -1)), -1)
        hidden, _ = self.lstm(inputs)
        raw = self.readout(hidden).squeeze(-1).transpose(0, 1)
        return functional.softplus(raw) if self.config.output_activation == "softplus" else raw
