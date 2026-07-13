"""Neural replacements for HBV process slots."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import torch
from torch import nn

from .processes import SoilProcess, compute_actual_et, update_soil_moisture


class NeuralRecharge(SoilProcess):
    """HBV soil process with an MLP recharge relation and physical storage/ET."""

    introduces: ClassVar[dict[str, tuple[float, float]]] = {
        "fc": (50.0, 700.0),
        "lp": (0.3, 1.0),
    }

    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        soil_input: torch.Tensor,
        pet: torch.Tensor,
        soil_moisture: torch.Tensor,
        parameters: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fc = parameters["fc"]
        safe_fc = torch.where(fc > 0.0, fc, torch.ones_like(fc))
        features = torch.stack(
            (
                soil_input / safe_fc,
                soil_moisture / safe_fc,
                pet / safe_fc,
            ),
            dim=-1,
        )
        recharge_fraction = self.mlp(features).squeeze(-1)
        recharge = torch.where(
            (fc > 0.0) & (soil_input > 0.0),
            soil_input * recharge_fraction,
            torch.zeros_like(soil_input),
        )
        actual_et = compute_actual_et(pet, soil_moisture, fc, parameters["lp"])
        new_soil_moisture, overflow = update_soil_moisture(
            soil_moisture,
            soil_input,
            recharge,
            actual_et,
            fc,
        )
        recharge_total = recharge + overflow
        return new_soil_moisture, recharge_total, actual_et
