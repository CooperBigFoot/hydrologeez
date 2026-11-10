"""One NPZ parameter archive and resident CPU inputs joined to prepared attributes."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from hydrologeez.camels import CamelsAttributes

from .model import ModelConfig
from .runtime import Batch


def save_examples(
    path: str | Path,
    values: NDArray[np.float32] | NDArray[np.float64],
    *,
    ids: Sequence[str],
    parameters: Sequence[str],
    units: Sequence[str],
) -> None:
    """Write physical values unchanged, with one catchment ID per example.

    The uncompressed NPZ holds version, values [example, parameter], ids,
    parameter names and units. It never stores attributes or fits preprocessing.
    Multiple rows per catchment need not have equal counts. Row positions identify
    examples within this archive. Float32/float64 precision is retained on disk.
    """
    if values.ndim != 2 or values.shape != (len(ids), len(parameters)) or not len(ids):
        raise ValueError("Values must be nonempty [example, parameter] with matching IDs")
    if not parameters or len(set(parameters)) != len(parameters) or len(units) != len(parameters):
        raise ValueError("Select unique parameter names and matching units")
    if any(not isinstance(gauge, str) or not gauge for gauge in ids):
        raise ValueError("Catchment IDs must be nonempty strings")
    if values.dtype not in (np.dtype("float32"), np.dtype("float64")) or not np.isfinite(values).all():
        raise ValueError("Values must be finite float32 or float64 physical parameters")
    with Path(path).open("wb") as output:
        np.savez(
            output,
            version=np.array(1),
            values=values,
            ids=np.array(ids),
            parameters=np.array(parameters),
            units=np.array(units),
        )


@dataclass(frozen=True)
class Inputs:
    """Selected resident pool. Attributes are stored once per selected catchment.

    All tensors remain on CPU. Do not mutate shared slices in place. IDs follow
    example order; catchments follow caller selection order. Attribute scaling
    belongs to the caller and must stay fixed during evaluation and sampling.
    """

    ids: tuple[str, ...]
    catchments: tuple[str, ...]
    parameter_names: tuple[str, ...]
    units: tuple[str, ...]
    attribute_names: tuple[str, ...]
    parameters: Tensor
    attributes: Tensor
    catchment_indices: Tensor

    def batch(self, config: ModelConfig, *, examples: slice | list[int] = slice(None), device: str = "cpu") -> Batch:
        """Pair selected examples with static inputs; transfer only this batch."""
        if (config.parameters, config.units, config.attributes) != (
            self.parameter_names,
            self.units,
            self.attribute_names,
        ):
            raise ValueError("Model parameter/unit/attribute order differs from selected inputs")
        parameters = self.parameters[examples]
        if len(parameters) == 0:
            raise ValueError("Batch must contain examples")
        return Batch(parameters.to(device), self.attributes[self.catchment_indices[examples]].to(device))


def load_inputs(
    path: str | Path,
    ids: Sequence[str],
    *,
    parameters: Sequence[str],
    attributes: CamelsAttributes,
    attribute_names: Sequence[str],
) -> Inputs:
    """Select catchments once; preserve stored example order and selected columns.

    Supply distinct training/validation catchment pools explicitly. For row-level
    splits, select batch indices explicitly. No shuffle, scaling, clipping,
    categorical encoding or missing-value imputation occurs here.
    """
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Select unique nonempty catchment IDs")
    if (
        not parameters
        or len(set(parameters)) != len(parameters)
        or not attribute_names
        or len(set(attribute_names)) != len(attribute_names)
    ):
        raise ValueError("Select unique nonempty parameter and attribute names")
    with np.load(path, allow_pickle=False) as archive:
        if archive["version"].item() != 1:
            raise ValueError("Unsupported parameter archive version")
        stored_ids = archive["ids"].tolist()
        names = archive["parameters"].tolist()
        units = archive["units"].tolist()
        if not set(ids) <= set(stored_ids) or not set(parameters) <= set(names):
            raise ValueError("Unknown catchment or parameter")
        selected_ids = set(ids)
        rows = [i for i, gauge in enumerate(stored_ids) if gauge in selected_ids]
        columns = [names.index(name) for name in parameters]
        values = archive["values"][np.ix_(rows, columns)]
    table = attributes.table
    positions = {gauge: i for i, gauge in enumerate(table["gauge_id"].to_list())}
    if table["gauge_id"].n_unique() != len(table):
        raise ValueError("Attribute catchment IDs must be unique")
    if not set(ids) <= positions.keys() or not set(attribute_names) <= set(table.columns):
        raise ValueError("Unknown attribute or catchment")
    if any(not table.schema[name].is_numeric() for name in attribute_names):
        raise ValueError("Select numeric attributes; categorical encoding belongs to caller")
    static = table[[positions[gauge] for gauge in ids]].select(attribute_names).to_numpy().astype(values.dtype)
    if not np.isfinite(static).all():
        raise ValueError("Selected attributes must be finite; no imputation is applied")
    catchments = {gauge: i for i, gauge in enumerate(ids)}
    example_ids = tuple(stored_ids[i] for i in rows)
    return Inputs(
        example_ids,
        tuple(ids),
        tuple(parameters),
        tuple(units[i] for i in columns),
        tuple(attribute_names),
        torch.from_numpy(np.ascontiguousarray(values)),
        torch.from_numpy(static),
        torch.tensor([catchments[gauge] for gauge in example_ids], dtype=torch.long),
    )
