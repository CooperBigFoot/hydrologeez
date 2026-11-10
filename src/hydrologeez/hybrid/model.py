"""Shared meteorological encoder, static parameter head and pretext head."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .data import Normalization, prepare
from .encoders import KDA, LSTM
from .hbv import Memory, Result, simulate


@dataclass(frozen=True)
class ModelConfig:
    encoder: Literal["lstm", "kda"] = "lstm"
    features: tuple[str, ...] = ("P", "T", "PET")
    attributes: tuple[str, ...] = ()
    hidden: int = 16
    routing_support: int = 7


# Baseline search ranges, not universal hydrological calibration bounds.
# TT CFMAX CFR CWH SCF FC BETA LP PERC Alpha K1 K2 MAXBAS
PARAMETER_NAMES = ("TT", "CFMAX", "CFR", "CWH", "SCF", "FC", "BETA", "LP", "PERC", "Alpha", "K1", "K2", "MAXBAS")
LOWER = (-2.0, 0.1, 0.01, 0.01, 0.5, 10.0, 0.3, 0.1, 0.01, 0.01, 0.001, 0.001, 1.0)
UPPER = (2.0, 5.0, 0.2, 0.2, 1.5, 500.0, 3.0, 1.0, 4.0, 0.5, 0.2, 0.1, 7.0)


class HybridHBV(nn.Module):
    """One fixed parameter vector per basin from its full supplied context.

    The recurrent representation is causal. The last valid representation uses
    the entire context window; applying its parameters to that same window is
    offline estimation, not a causal discharge forecast. Zone parameters are
    shared within a basin; each zone retains separate physical water stores.
    """

    mean: Tensor
    scale: Tensor
    parameter_lower: Tensor
    parameter_upper: Tensor

    def __init__(self, config: ModelConfig, normalization: Normalization) -> None:
        super().__init__()
        if config.encoder not in ("lstm", "kda") or config.hidden < 1 or config.routing_support < 2:
            raise ValueError("invalid encoder, hidden size or routing support")
        if config.features != normalization.features or len(set(config.attributes)) != len(config.attributes):
            raise ValueError("feature/attribute order must be explicit and unique")
        self.config = config
        self.register_buffer("mean", normalization.mean.detach().clone())
        self.register_buffer("scale", normalization.scale.detach().clone())
        self.register_buffer("parameter_lower", torch.tensor(LOWER))
        upper = torch.tensor(UPPER)
        upper[-1] = config.routing_support
        self.register_buffer("parameter_upper", upper)
        inputs = 3 * len(config.features) + len(config.attributes)
        self.encoder = LSTM(inputs, config.hidden) if config.encoder == "lstm" else KDA(inputs, config.hidden)
        self.parameter_head = nn.Sequential(
            nn.Linear(config.hidden, config.hidden), nn.SiLU(), nn.Linear(config.hidden, 13)
        )
        self.pretext_head = nn.Sequential(
            nn.Linear(config.hidden, config.hidden), nn.SiLU(), nn.Linear(config.hidden, len(config.features))
        )
        self.frozen_encoder = False

    @property
    def normalization(self) -> Normalization:
        return Normalization(self.mean, self.scale, self.config.features)

    def freeze_encoder(self, frozen: bool) -> None:
        self.frozen_encoder = frozen
        self.encoder.requires_grad_(not frozen)
        self.encoder.train(self.training and not frozen)

    def train(self, mode: bool = True) -> "HybridHBV":
        super().train(mode)
        if self.frozen_encoder:
            self.encoder.eval()
        return self

    def represent(self, values: Tensor, attributes: Tensor, padding: Tensor, hidden: Tensor | None = None) -> Tensor:
        if attributes.shape != (values.shape[0], len(self.config.attributes)) or not bool(
            torch.isfinite(attributes).all()
        ):
            raise ValueError("attributes must be finite in configured order")
        if (
            padding.shape != values.shape[:2]
            or bool(padding[:, 0].any())
            or bool((padding[:, :-1] & ~padding[:, 1:]).any())
        ):
            raise ValueError("only right padding of nonempty sequences is supported")
        prepared = prepare(
            values,
            padding,
            torch.zeros_like(values, dtype=torch.bool) if hidden is None else hidden,
            self.normalization,
        )
        inputs = torch.cat((prepared.inputs, attributes[:, None, :].expand(-1, values.shape[1], -1)), -1)
        return self.encoder(inputs)

    def physical_parameters(self, representations: Tensor, padding: Tensor) -> Tensor:
        last = (~padding).sum(-1) - 1
        context = representations[torch.arange(len(last), device=last.device), last]
        return self.parameter_lower + self.parameter_head(context).sigmoid() * (
            self.parameter_upper - self.parameter_lower
        )

    def forward(  # ty: ignore[invalid-mutable-override] -- PyTorch declares forward as writable Callable.
        self,
        values: Tensor,
        attributes: Tensor,
        padding: Tensor,
        forcing: Tensor,
        area: Tensor,
        initial: Memory | None = None,
    ) -> Result:
        parameters = self.physical_parameters(self.represent(values, attributes, padding), padding)
        zones = parameters[:, None, :8].expand(-1, forcing.shape[2], -1)
        return simulate(
            forcing, area, zones, parameters[:, 8:12], parameters[:, 12], initial, support=self.config.routing_support
        )
