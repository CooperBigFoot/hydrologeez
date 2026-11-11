"""Reusable training stages, validation, inference and checkpoint persistence."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from hydrologeez.losses import masked_mse, soft_npe_loss

from .data import Normalization, prepare
from .hbv import Memory, Result
from .model import HybridHBV, ModelConfig

Stage = Literal["direct", "pretrain", "frozen"]


@dataclass(frozen=True)
class Batch:
    meteorology: Tensor  # [basin, context day, feature], may contain missing NaN
    attributes: Tensor  # [basin, attribute], caller-scaled
    padding: Tensor  # [basin, context day], true only for right padding
    forcing: Tensor  # [simulation day, basin, zone, P/T/PET], physical units
    area: Tensor  # [basin, zone]
    discharge: Tensor | None = None  # [simulation day, basin]
    discharge_valid: Tensor | None = None
    initial: Memory | None = None


class Training:
    """Adam updates for direct discharge, reconstruction, or frozen discharge.

    Each step owns a complete supplied batch/window. No implicit state is carried
    between batches. Stage changes reset optimizer moments; checkpoints resume
    the current stage's optimizer. Caller owns batching and epoch selection.
    """

    def __init__(
        self,
        model: HybridHBV,
        *,
        stage: Stage = "direct",
        learning_rate: float = 0.001,
        discharge_loss: Literal["mse", "soft_npe"] = "mse",
        softness: float = 0.1,
    ) -> None:
        if discharge_loss not in ("mse", "soft_npe"):
            raise ValueError("unknown discharge loss")
        if not math.isfinite(softness) or softness <= 0:
            raise ValueError("softness must be finite and positive")
        self.discharge_loss = discharge_loss
        self.softness = softness
        self.model = model
        self.learning_rate = learning_rate
        self.set_stage(stage)

    def set_stage(self, stage: Stage) -> None:
        if stage not in ("direct", "pretrain", "frozen"):
            raise ValueError("unknown training stage")
        self.stage = stage
        self.model.freeze_encoder(stage == "frozen")
        self.model.parameter_head.requires_grad_(stage != "pretrain")
        self.model.pretext_head.requires_grad_(stage == "pretrain")
        self.model.zero_grad(set_to_none=True)
        self.optimizer = torch.optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad), lr=self.learning_rate
        )

    def loss(self, batch: Batch, *, hidden: Tensor | None = None) -> Tensor:
        if self.stage == "pretrain":
            if hidden is None:
                raise ValueError("pretraining needs an explicit synthetic hidden mask")
            prepared = prepare(batch.meteorology, batch.padding, hidden, self.model.normalization)
            prediction = self.model.pretext_head(
                self.model.represent(batch.meteorology, batch.attributes, batch.padding, hidden)
            )
            return masked_mse(prediction, prepared.normalized, prepared.targets)
        if batch.discharge is None or batch.discharge_valid is None:
            raise ValueError("discharge training needs observations and validity mask")
        prediction = self.model(
            batch.meteorology, batch.attributes, batch.padding, batch.forcing, batch.area, batch.initial
        )
        if self.discharge_loss == "soft_npe":
            return soft_npe_loss(prediction.outlet, batch.discharge, batch.discharge_valid, softness=self.softness)
        valid = batch.discharge_valid & torch.isfinite(batch.discharge)
        return masked_mse(prediction.outlet, batch.discharge, valid)

    def train_step(self, batch: Batch, *, hidden: Tensor | None = None) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss(batch, hidden=hidden)
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite training loss")
        loss.backward()
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise ValueError("missing or nonfinite gradients")
        self.optimizer.step()
        return float(loss.detach())

    @torch.no_grad()
    def evaluate(self, batch: Batch, *, hidden: Tensor | None = None) -> float:
        self.model.eval()
        return float(self.loss(batch, hidden=hidden))

    @torch.no_grad()
    def predict(self, batch: Batch) -> Result:
        self.model.eval()
        return self.model(batch.meteorology, batch.attributes, batch.padding, batch.forcing, batch.area, batch.initial)

    def save(self, path: str | Path) -> None:
        """Persist architecture, ordered features, scaling, bounds, stage and Adam.

        HBV water stores belong to the caller's simulation, not model weights;
        predict returns explicit final Memory for continuation with fixed params.
        """
        torch.save(
            {
                "version": 2,
                "discharge_loss": self.discharge_loss,
                "softness": self.softness,
                "config": asdict(self.model.config),
                "model": self.model.state_dict(),
                "stage": self.stage,
                "learning_rate": self.learning_rate,
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "Training":
        saved = torch.load(path, map_location=device, weights_only=True)
        if saved["version"] not in (1, 2):
            raise ValueError("unsupported checkpoint version")
        config = ModelConfig(**saved["config"])
        state = saved["model"]
        normalization = Normalization(state["mean"], state["scale"], config.features)
        model = HybridHBV(config, normalization).to(device=device, dtype=state["mean"].dtype)
        model.load_state_dict(state)
        training = cls(
            model,
            stage=saved["stage"],
            learning_rate=saved["learning_rate"],
            discharge_loss=saved["discharge_loss"] if saved["version"] == 2 else "mse",
            softness=saved["softness"] if saved["version"] == 2 else 0.1,
        )
        training.optimizer.load_state_dict(saved["optimizer"])
        return training
