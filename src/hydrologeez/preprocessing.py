"""Experiment preprocessing: exclusions, fitted transforms and persisted prepared datasets.

Basins are always the last axis: daily fields are [day, basin], static fields
are [basin]. Fitting happens only in ``fit``, on accepted values inside an
explicitly authorized pool; applying or loading a ``Transform`` never refits.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Values = NDArray[np.floating]
Mask = NDArray[np.bool_]
Scope = Literal["global", "basin"]


def standardization(values: Array, selected: Mask, axis: int | tuple[int, ...]) -> tuple[Array, Array]:
    """Population mean and standard deviation of selected values along axis.

    A constant selection gets scale 1, so its standardized values are 0.
    """
    count = selected.sum(axis, keepdims=True)
    if bool((count == 0).any()):
        raise ValueError("Each feature needs selected values in its fitting pool")
    mean = np.where(selected, values, 0.0).sum(axis, keepdims=True) / count
    scale = np.sqrt((np.where(selected, values - mean, 0.0) ** 2).sum(axis, keepdims=True) / count)
    return np.squeeze(mean, axis), np.squeeze(np.where(scale > 0, scale, 1.0), axis)


@dataclass(frozen=True)
class Field:
    """One source field and the values the experiment accepts as observed.

    ``accepted`` is false for source-missing values and for observations the
    experiment excludes; accepted values must be finite.
    """

    values: Values
    accepted: Mask
    units: str

    def __post_init__(self) -> None:
        if self.values.ndim not in (1, 2) or self.accepted.shape != self.values.shape:
            raise ValueError("Field values and accepted mask must share a [basin] or [day, basin] shape")
        if self.accepted.dtype != np.bool_ or not np.isfinite(self.values[self.accepted]).all():
            raise ValueError("Accepted field values must be finite with a boolean mask")


@dataclass(frozen=True)
class Exclusion:
    """Caller-chosen named QC evidence that makes observations unavailable.

    Daily reasons are boolean [day, basin] arrays; basin assessments are
    boolean [basin] arrays that exclude every observation of a flagged basin.
    """

    daily: tuple[str, ...] = ()
    basin: tuple[str, ...] = ()

    def mask(self, daily: Mapping[str, Mask], basin: Mapping[str, Mask]) -> Mask:
        """Excluded observations as [day, basin]; the evidence is not modified."""
        if not set(self.daily) <= daily.keys() or not set(self.basin) <= basin.keys():
            raise ValueError("Exclusion names unknown QC evidence")
        shapes = {array.shape for array in daily.values()}
        if len(shapes) != 1 or len(shape := shapes.pop()) != 2:
            raise ValueError("Daily QC evidence must share one [day, basin] shape")
        if any(array.shape != shape[1:] for array in basin.values()):
            raise ValueError("Basin assessments must align with daily evidence basins")
        excluded = np.zeros(shape, dtype=np.bool_)
        for name in self.daily:
            excluded |= daily[name]
        for name in self.basin:
            excluded |= basin[name][None, :]
        return excluded


@dataclass(frozen=True)
class Log:
    """Natural logarithm of value + offset; the offset is expressed in ``units``."""

    offset: float
    units: str


@dataclass(frozen=True)
class ZScore:
    """Standardize with one parameter set per feature (global) or per basin."""

    scope: Scope = "global"


@dataclass(frozen=True)
class Standardization:
    """Fitted z-score parameters; ``ids`` lists the fitted basins for basin scope."""

    scope: Scope
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    count: tuple[int, ...]
    ids: tuple[str, ...] = ()

    def _parameters(self, ids: Sequence[str]) -> tuple[Array, Array]:
        if self.scope == "global":
            return np.array(self.mean), np.array(self.scale)
        positions = {gauge: i for i, gauge in enumerate(self.ids)}
        unseen = [gauge for gauge in ids if gauge not in positions]
        if unseen:
            raise ValueError(f"No per-basin transform was fitted for basins: {', '.join(unseen)}")
        index = [positions[gauge] for gauge in ids]
        return np.array(self.mean)[index], np.array(self.scale)[index]


Step = Log | Standardization


@dataclass(frozen=True)
class Transform:
    """A fitted chain for one field, applied in order and inverted in reverse."""

    units: str
    steps: tuple[Step, ...] = ()

    def forward(self, values: Values, ids: Sequence[str]) -> Array:
        """Transform [basin] or [day, basin] values; nonfinite values stay nonfinite."""
        if values.shape[-1] != len(ids):
            raise ValueError("Values need one trailing basin axis matching ids")
        result = np.asarray(values, dtype=np.float64)
        for step in self.steps:
            if isinstance(step, Log):
                shifted = result + step.offset
                if bool((np.isfinite(shifted) & (shifted <= 0)).any()):
                    raise ValueError("log needs value + offset > 0 for every accepted value")
                result = np.log(shifted)
            else:
                mean, scale = step._parameters(ids)
                result = (result - mean) / scale
        return result

    def inverse(self, values: Values, ids: Sequence[str]) -> Array:
        """Undo forward for [basin] or [day, basin] values."""
        if values.shape[-1] != len(ids):
            raise ValueError("Values need one trailing basin axis matching ids")
        result = np.asarray(values, dtype=np.float64)
        for step in reversed(self.steps):
            if isinstance(step, Log):
                result = np.exp(result) - step.offset
            else:
                mean, scale = step._parameters(ids)
                result = result * scale + mean
        return result

    def to_json(self) -> dict[str, Any]:
        steps: list[dict[str, Any]] = [
            {"kind": "log", "offset": step.offset, "units": step.units}
            if isinstance(step, Log)
            else {
                "kind": "zscore",
                "scope": step.scope,
                "ids": list(step.ids),
                "mean": list(step.mean),
                "scale": list(step.scale),
                "count": list(step.count),
            }
            for step in self.steps
        ]
        return {"units": self.units, "steps": steps}

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "Transform":
        steps: list[Step] = []
        for step in payload["steps"]:
            if step["kind"] == "log":
                steps.append(Log(step["offset"], step["units"]))
            elif step["kind"] == "zscore":
                steps.append(
                    Standardization(
                        step["scope"],
                        tuple(step["mean"]),
                        tuple(step["scale"]),
                        tuple(step["count"]),
                        tuple(step["ids"]),
                    )
                )
            else:
                raise ValueError(f"Unsupported transform step: {step['kind']}")
        return cls(payload["units"], tuple(steps))


def fit(chain: Sequence[Log | ZScore], source: Field, ids: Sequence[str], pool: Mask) -> Transform:
    """Fit a transform chain on accepted values inside an explicitly authorized pool.

    The pool has the field's shape. A static [basin] field also accepts a
    [day, basin] pool; a basin is authorized when any of its days is, and it
    counts once. Basin scope needs daily values and fits only basins with
    pooled values; other basins cannot be transformed by it.
    """
    if len(ids) != source.values.shape[-1] or len(set(ids)) != len(ids):
        raise ValueError("ids must be unique and match the field's basin axis")
    if source.values.ndim == 1 and pool.ndim == 2:
        pool = pool.any(axis=0)
    if pool.shape != source.values.shape or pool.dtype != np.bool_:
        raise ValueError("The fitting pool must be a boolean mask aligned to the field")
    selected = source.accepted & pool
    if not selected.any():
        raise ValueError("The fitting pool contains no accepted values")
    current = np.where(selected, source.values.astype(np.float64), np.nan)
    steps: list[Step] = []
    for step in chain:
        if isinstance(step, Log):
            if step.units != source.units:
                raise ValueError(f"log offset units {step.units} differ from field units {source.units}")
            fitted: Step = step
        elif step.scope == "global":
            mean, scale = standardization(current, selected, axis=tuple(range(current.ndim)))
            fitted = Standardization("global", (float(mean),), (float(scale),), (int(selected.sum()),))
        elif step.scope == "basin":
            if current.ndim != 2:
                raise ValueError("Per-basin scope needs daily [day, basin] values")
            fitted_basins = selected.any(axis=0)
            mean, scale = standardization(current[:, fitted_basins], selected[:, fitted_basins], axis=0)
            fitted = Standardization(
                "basin",
                tuple(mean.tolist()),
                tuple(scale.tolist()),
                tuple(selected[:, fitted_basins].sum(axis=0).tolist()),
                tuple(gauge for gauge, kept in zip(ids, fitted_basins, strict=True) if kept),
            )
        else:
            raise ValueError(f"Unsupported z-score scope: {step.scope}")
        steps.append(fitted)
        # Unpooled basins hold no selected values; only transform pooled ones.
        basins = selected.reshape(-1, len(ids)).any(axis=0)
        current[..., basins] = Transform(source.units, (fitted,)).forward(
            current[..., basins], [gauge for gauge, kept in zip(ids, basins, strict=True) if kept]
        )
    return Transform(source.units, tuple(steps))


@dataclass(frozen=True)
class PhysicalForcing:
    """Untransformed precipitation, temperature and PET [day, basin, 3] for HBV."""

    names: tuple[str, str, str]
    units: tuple[str, str, str]
    values: Values


def _model_space(source: Field, transform: Transform, ids: Sequence[str], fill_value: float) -> tuple[Array, Mask]:
    if transform.units != source.units:
        raise ValueError(f"Transform units {transform.units} differ from field units {source.units}")
    transformed = transform.forward(np.where(source.accepted, source.values, np.nan), ids)
    filled = ~source.accepted
    return np.where(filled, fill_value, transformed), filled


def prepare(
    output: Path,
    *,
    ids: Sequence[str],
    dates: NDArray[np.int64],
    inputs: Mapping[str, Field],
    attributes: Mapping[str, Field],
    transforms: Mapping[str, Transform],
    forcing: PhysicalForcing,
    discharge: Field | None = None,
    discharge_transform: Transform | None = None,
    fill_value: float = 0.0,
    recipe: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Write transformed, filled model inputs and targets once, with their recipe.

    Daily inputs and discharge are [day, basin] over consecutive Gregorian
    ordinals; attributes are [basin]. Every input and attribute needs a fitted
    transform (``Transform(units)`` keeps values unchanged). Values the field
    does not accept are replaced by ``fill_value`` after transforming and
    flagged in ``<field>_was_filled``. Discharge keeps physical values, a
    transformed target that is NaN where ineligible, and the eligibility mask.
    Physical forcing must be valid; it is stored unchanged and never filled.
    ``recipe`` and ``provenance`` are caller JSON describing selections,
    exclusions, fitting pools and sources. The output directory must not exist;
    metadata is written last as the completion marker.
    """
    days, basins = len(dates), len(ids)
    if not basins or len(set(ids)) != basins or not days or np.any(np.diff(dates) != 1):
        raise ValueError("Select unique basins and consecutive daily dates")
    if not inputs or set(inputs) & set(attributes) or transforms.keys() != inputs.keys() | attributes.keys():
        raise ValueError("Name each input and attribute once and give each a fitted transform")
    if any(item.values.shape != (days, basins) for item in inputs.values()) or any(
        item.values.shape != (basins,) for item in attributes.values()
    ):
        raise ValueError("Inputs must be [day, basin] and attributes [basin] in selected order")
    if (discharge is None) != (discharge_transform is None):
        raise ValueError("Supply discharge together with its fitted transform")
    if forcing.values.shape != (days, basins, 3) or not np.isfinite(forcing.values).all():
        raise ValueError("Physical forcing must be finite [day, basin, P/T/PET]; no imputation is applied")
    if np.any(forcing.values[:, :, (0, 2)] < 0):
        raise ValueError("Physical precipitation and PET must be nonnegative")
    dynamic = [_model_space(item, transforms[name], ids, fill_value) for name, item in inputs.items()]
    static = [_model_space(item, transforms[name], ids, fill_value) for name, item in attributes.items()]
    target = target_metadata = None
    if discharge is not None and discharge_transform is not None:
        if discharge.values.shape != (days, basins) or discharge_transform.units != discharge.units:
            raise ValueError("Discharge must be [day, basin] in its transform's units")
        target = discharge_transform.forward(np.where(discharge.accepted, discharge.values, np.nan), ids)
        target_metadata = {"units": discharge.units, "transform": discharge_transform.to_json()}
    metadata = {
        "format": 1,
        "ids": list(ids),
        "start": int(dates[0]),
        "stop": int(dates[-1] + 1),
        "calendar": "Gregorian ordinals; stop exclusive",
        "fill_value": fill_value,
        "inputs": [{"name": name, "transform": transforms[name].to_json()} for name in inputs],
        "attributes": [{"name": name, "transform": transforms[name].to_json()} for name in attributes],
        "forcing": {"names": list(forcing.names), "units": list(forcing.units)},
        "discharge": target_metadata,
        "recipe": recipe,
        "provenance": provenance,
    }
    # Validate serializability before creating the output directory.
    text = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "inputs.npy", np.stack([values.T for values, _ in dynamic], -1).astype(np.float32))
    np.save(output / "inputs_was_filled.npy", np.stack([filled.T for _, filled in dynamic], -1))
    np.save(output / "attributes.npy", np.array([v for v, _ in static], dtype=np.float32).T.reshape(basins, -1))
    np.save(
        output / "attributes_was_filled.npy", np.array([f for _, f in static], dtype=np.bool_).T.reshape(basins, -1)
    )
    np.save(output / "forcing.npy", forcing.values)
    if discharge is not None and target is not None:
        np.save(output / "discharge.npy", discharge.values)
        np.save(output / "target.npy", target)
        np.save(output / "target_eligible.npy", discharge.accepted)
    (output / "prepared.json").write_text(text)


FILLED_SUFFIX = "_was_filled"


@dataclass(frozen=True)
class PreparedDischarge:
    """Physical discharge, its transformed target and target eligibility, all [day, basin].

    The target is NaN where ineligible; only eligible targets may enter loss or evaluation.
    """

    units: str
    physical: Array
    target: Array
    eligible: Mask
    transform: Transform


@dataclass(frozen=True)
class PreparedDataset:
    """Resident prepared experiment data, read without fitting or transforming.

    Inputs are [basin, day, feature] and attributes [basin, attribute] in final
    model-input space; filled flags share their shapes. Forcing is physical
    [day, basin, P/T/PET].
    """

    directory: Path
    ids: tuple[str, ...]
    dates: NDArray[np.int64]
    features: tuple[str, ...]
    units: tuple[str, ...]
    inputs: NDArray[np.float32]
    inputs_was_filled: Mask
    attribute_names: tuple[str, ...]
    attribute_units: tuple[str, ...]
    attributes: NDArray[np.float32]
    attributes_was_filled: Mask
    transforms: dict[str, Transform]
    physical: tuple[str, str, str]
    physical_units: tuple[str, str, str]
    forcing: Values
    discharge: PreparedDischarge | None
    fill_value: float
    recipe: dict[str, Any]
    provenance: dict[str, Any]

    @classmethod
    def read(cls, directory: Path) -> "PreparedDataset":
        metadata = json.loads((directory / "prepared.json").read_text())
        if metadata["format"] != 1:
            raise ValueError("Unsupported prepared dataset format")
        transforms = {
            item["name"]: Transform.from_json(item["transform"])
            for item in (*metadata["inputs"], *metadata["attributes"])
        }
        features = tuple(item["name"] for item in metadata["inputs"])
        attribute_names = tuple(item["name"] for item in metadata["attributes"])
        target = metadata["discharge"]

        def load(name: str) -> Any:
            return np.load(directory / f"{name}.npy", allow_pickle=False)

        return cls(
            directory,
            tuple(metadata["ids"]),
            np.arange(metadata["start"], metadata["stop"], dtype=np.int64),
            features,
            tuple(transforms[name].units for name in features),
            load("inputs"),
            load("inputs_was_filled"),
            attribute_names,
            tuple(transforms[name].units for name in attribute_names),
            load("attributes"),
            load("attributes_was_filled"),
            transforms,
            tuple(metadata["forcing"]["names"]),
            tuple(metadata["forcing"]["units"]),
            load("forcing"),
            None
            if target is None
            else PreparedDischarge(
                target["units"],
                load("discharge"),
                load("target"),
                load("target_eligible"),
                Transform.from_json(target["transform"]),
            ),
            metadata["fill_value"],
            metadata["recipe"],
            metadata["provenance"],
        )

    def model_inputs(self, names: Sequence[str]) -> NDArray[np.float32]:
        """Select [basin, day, name] columns; ``<field>_was_filled`` names select flags as 0/1."""
        return np.stack([self._column(name, self.features, self.inputs, self.inputs_was_filled) for name in names], -1)

    def model_attributes(self, names: Sequence[str]) -> NDArray[np.float32]:
        """Select [basin, name] columns; ``<field>_was_filled`` names select flags as 0/1."""
        columns = [
            self._column(name, self.attribute_names, self.attributes, self.attributes_was_filled) for name in names
        ]
        return np.stack(columns, -1) if columns else np.empty((len(self.ids), 0), dtype=np.float32)

    @staticmethod
    def _column(name: str, fields: tuple[str, ...], values: NDArray[np.float32], filled: Mask) -> NDArray[np.float32]:
        if name in fields:
            return values[..., fields.index(name)]
        base = name.removesuffix(FILLED_SUFFIX)
        if name.endswith(FILLED_SUFFIX) and base in fields:
            return filled[..., fields.index(base)].astype(np.float32)
        raise ValueError(f"Unknown prepared field: {name}")
