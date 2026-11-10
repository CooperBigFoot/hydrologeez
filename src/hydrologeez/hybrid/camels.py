"""Resident CAMELS tensors with explicit feature, physical and observation pools."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from hydrologeez.camels import Camels, CamelsAttributes, Observations
from hydrologeez.camels_meteorology import CamelsMeteorology, Meteorology

from .model import ModelConfig
from .runtime import Batch


@dataclass(frozen=True)
class CamelsInputs:
    """Load once on CPU, slice repeatedly. No scaling or synthetic masking is fitted.

    Slices share resident storage; advanced basin indexing and device transfers
    copy only the requested batch. Do not mutate shared tensors in place.
    """

    source: Meteorology
    features: tuple[str, ...]
    attribute_names: tuple[str, ...]
    meteorology: Tensor
    attributes: Tensor
    forcing: Tensor
    observations: Observations | None
    discharge: Tensor | None
    discharge_valid: Tensor | None

    def batch(
        self,
        config: ModelConfig,
        *,
        basins: slice | list[int] = slice(None),
        context: slice = slice(None),
        simulation: slice = slice(None),
        evaluation_mask: Tensor | None = None,
        device: str = "cpu",
    ) -> Batch:
        """Build ordered runtime inputs, allowing separate context/simulation dates.

        evaluation_mask is in returned simulation-day/basin order and can only
        remove valid targets. Initial physical memory remains a caller choice.
        """
        if config.features != self.features or config.attributes != self.attribute_names:
            raise ValueError("Model feature/attribute order differs from selected inputs")
        meteorology = self.meteorology[basins, context].to(device)
        attributes = self.attributes[basins].to(device)
        forcing = self.forcing[simulation, basins].to(device)
        if meteorology.shape[0] == 0 or meteorology.shape[1] == 0 or forcing.shape[0] == 0:
            raise ValueError("Batch must contain basins, context and simulation days")
        valid = None if self.discharge_valid is None else self.discharge_valid[simulation, basins].to(device)
        if evaluation_mask is not None:
            if valid is None or evaluation_mask.shape != valid.shape or evaluation_mask.dtype != torch.bool:
                raise ValueError("Evaluation mask needs matching authorized discharge")
            valid = valid & evaluation_mask.to(device)
        return Batch(
            meteorology,
            attributes,
            torch.zeros(meteorology.shape[:2], dtype=torch.bool, device=device),
            forcing,
            torch.ones((len(meteorology), 1), device=device),
            None if self.discharge is None else self.discharge[simulation, basins].to(device),
            valid,
        )


def load_inputs(
    store: CamelsMeteorology,
    ids: Sequence[str],
    start: str,
    stop: str,
    *,
    features: Sequence[str],
    physical: tuple[str, str, str],
    attributes: CamelsAttributes | None = None,
    attribute_names: Sequence[str] = (),
    observations: Camels | None = None,
) -> CamelsInputs:
    """Load an explicitly authorized basin/date pool, without fitting preprocessing.

    Physical names must be product.prcp, product.tmean, product.pet in that order;
    products may differ. Discharge is not read unless observations is supplied.
    Static fields must be explicitly chosen numeric finite columns. Categorical
    encoding, missing-attribute handling and attribute scaling belong to callers.
    """
    if not features or len(set(features)) != len(features):
        raise ValueError("Select unique nonempty neural features")
    if len(physical) != 3 or tuple(name.split(".")[-1] for name in physical) != ("prcp", "tmean", "pet"):
        raise ValueError("Select physical precipitation, mean temperature and PET in order")
    selected = store.select(ids, start, stop, tuple(dict.fromkeys((*features, *physical))))
    columns = [selected.features.index(field) for field in features]
    physical_columns = [selected.features.index(field) for field in physical]
    if not selected.valid[:, :, physical_columns].all():
        raise ValueError("Invalid/missing physical forcing; no imputation is applied")
    neural = np.where(selected.valid[:, :, columns], selected.values[:, :, columns], np.nan)
    forcing = np.ascontiguousarray(selected.values[:, :, physical_columns].transpose(1, 0, 2)[:, :, None, :])
    if len(set(attribute_names)) != len(attribute_names):
        raise ValueError("Attribute names must be unique")
    static = np.empty((len(ids), 0), dtype=np.float32)
    if attribute_names:
        if attributes is None:
            raise ValueError("Selected attributes need an authorized attribute table")
        table = attributes.table
        positions = {gauge: i for i, gauge in enumerate(table["gauge_id"].to_list())}
        if not set(ids) <= positions.keys() or not set(attribute_names) <= set(table.columns):
            raise ValueError("Unknown attribute or basin")
        if any(not table.schema[field].is_numeric() for field in attribute_names):
            raise ValueError("Encode categorical attributes explicitly before runtime use")
        static = table[[positions[gauge] for gauge in ids]].select(attribute_names).to_numpy().astype(np.float32)
        if not np.isfinite(static).all():
            raise ValueError("Selected attributes must be finite; no imputation is applied")
    observed = None
    discharge = valid = None
    if observations is not None:
        n = len(selected.dates)
        q = np.full((n, len(ids)), np.nan)
        converted = np.full_like(q, np.nan)
        present = np.zeros(q.shape, dtype=np.bool_)
        source = [observations.observations(gauge) for gauge in ids]
        qc = np.full(q.shape, "", dtype=np.result_type(*(item.qc.dtype for item in source)))
        first, last = int(selected.dates[0]), int(selected.dates[-1] + 1)
        for i, item in enumerate(source):
            a, z = max(first, int(item.dates[0])), min(last, int(item.dates[-1] + 1))
            if z > a:
                src, dest = slice(a - int(item.dates[0]), z - int(item.dates[0])), slice(a - first, z - first)
                q[dest, i], converted[dest, i] = item.discharge_cfs[src], item.discharge_mm_day[src]
                present[dest, i], qc[dest, i] = item.present[src], item.qc[src]
        observed = Observations(selected.dates, q, converted, qc, present)
        discharge = torch.from_numpy(converted.astype(np.float32))
        valid = torch.from_numpy(present & np.isfinite(converted))
    return CamelsInputs(
        selected,
        tuple(features),
        tuple(attribute_names),
        torch.from_numpy(np.ascontiguousarray(neural)),
        torch.from_numpy(static),
        torch.from_numpy(forcing),
        observed,
        discharge,
        valid,
    )
