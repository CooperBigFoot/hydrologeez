# Experiment preprocessing

`hydrologeez.preprocessing` turns stored source data and QC evidence into a prepared dataset on disk. The workflow has two persisted stages:

1. The retrospective QC archive keeps original observations, daily QC reasons and basin/index assessments. It never enforces exclusions.
2. Experiment preparation selects basins, dates and fields, applies caller-chosen exclusions, fits transforms on authorized pools, fills missing inputs and writes the result once.

Model runs then read the prepared dataset repeatedly. Reading never fits, transforms or fills.

## Prepare once

Daily fields are `[day, basin]`; static fields are `[basin]`. A `Field` holds values, an `accepted` mask and units. `accepted` is false for source-missing values and for observations the experiment excludes.

```python
import numpy as np

from hydrologeez.preprocessing import (
    Exclusion,
    Field,
    Log,
    PhysicalForcing,
    Transform,
    ZScore,
    fit,
    homogeneity_flags,
    prepare,
)
from hydrologeez.quality_control import StreamflowQualityArchive

archive = StreamflowQualityArchive(qc_directory)
qc = archive.daily(ids, start, stop)  # reasons: name -> [day, basin]; discharge in source units
assessed = archive.homogeneity(ids, "yearly")
basin = {"mean_inhomogeneous": homogeneity_flags(assessed, ids, "MEAN", ("suspect", "doubtful"))}
rule = Exclusion(
    daily=("provider_suspect", "negative", "constant_run", "seasonal_outlier"), basin=("mean_inhomogeneous",)
)
excluded = rule.mask(qc.discharge.shape, qc.reasons, basin)
prcp = Field(meteorology.values[:, :, 0].T, meteorology.valid[:, :, 0].T, "mm/day")
area = Field(area_km2, np.isfinite(area_km2), "km2")
discharge = Field(q_mm_day, qc.observed & ~excluded, "mm/day")  # caller converts units explicitly

prepare(
    output,
    ids=ids,
    dates=meteorology.dates,
    inputs={"daymet.prcp": prcp},
    attributes={"area_gages2": area},
    transforms={
        "daymet.prcp": fit([Log(0.01, "mm/day"), ZScore()], prcp, ids, unlabeled_pool),
        "area_gages2": fit([ZScore()], area, ids, training_pool),  # each authorized basin counts once
    },
    forcing=PhysicalForcing(("daymet.prcp", "daymet.tmean", "daymet.pet"), ("mm/day", "degC", "mm/day"), physical),
    discharge=discharge,
    discharge_transform=fit([Log(0.01, "mm/day"), ZScore()], discharge, ids, fold_training_pool),
    recipe={"exclusion": {"daily": [...], "basin": [...]}, "pools": {"daymet.prcp": "...", "discharge": "..."}},
    provenance={"qc_archive": "...", "meteorology": "..."},
)
```

Rules:

- `fit` uses only accepted values inside the pool you pass. Pools are explicit boolean masks; nothing is fitted from other data.
- Chains run in the declared order. `Log(offset, units)` computes `log(value + offset)`; its units must match the field. `Transform.inverse` reverses the chain.
- `ZScore("global")` fits one mean and scale per field. `ZScore("basin")` fits one per basin and needs daily values. A basin without pooled values gets no per-basin parameters, so applying the transform to it raises.
- Constant pooled values get scale 1, so their standardized values are 0. Population standard deviation is used, as in `hybrid.data.Normalization`, which shares the same implementation.
- `prepared.json` stores fitted parameters and counts, not pool masks. The recipe must identify the fitting pool of every transform, such as basins and date ranges.
- Fit discharge transforms on the supervised training pool of each fold. Inputs may use another explicitly authorized pool, such as an unlabeled historical pool.
- After transforming, `prepare` replaces every non-accepted input with `fill_value` (default 0) and records `<field>_was_filled`. There is no interpolation or forward fill.
- Discharge keeps physical values, a transformed target that is NaN where ineligible, and an eligibility mask. Filled or excluded targets are never eligible.
- Physical P/T/PET must be valid. They are stored as float32, like `hybrid.camels.load_inputs`, and are never transformed or filled.

## Consume repeatedly

```python
from hydrologeez.hybrid.camels import prepared_inputs
from hydrologeez.hybrid.data import Normalization
from hydrologeez.hybrid.model import HybridHBV, ModelConfig
from hydrologeez.hybrid.runtime import Training
from hydrologeez.preprocessing import PreparedDataset

dataset = PreparedDataset.read(output)
assert dataset.discharge is not None
features = ("daymet.prcp", "daymet.prcp_was_filled")
inputs = prepared_inputs(dataset, features=features, attribute_names=("area_gages2",))
config = ModelConfig(features=features, attributes=("area_gages2",))
model = HybridHBV(config, Normalization.identity(features))
training = Training(model, discharge_transform=dataset.discharge.transform)
batch = inputs.batch(config)
```

Prepared inputs are already in final model-input space, so models use `Normalization.identity`. `batch.discharge` stays physical and `batch.discharge_valid` is the stored eligibility mask. `lstm.camels.batch_from_inputs` accepts the same `CamelsInputs`, and `lstm.runtime.Training` accepts the same `discharge_transform`.

Both `HybridHBV` and `DischargeLSTM` (`output_activation="softplus"`) emit nonnegative physical discharge in `mm/day` during `predict(batch)`, without silent clipping or double transformation. When `discharge_transform` is supplied to `Training`, `loss(batch)` and `evaluate(batch)` apply `discharge_transform.forward_tensor` differentiably to eligible predictions and targets and compute valid-element-weighted MSE in transformed discharge space. Checkpoints persist `discharge_transform` for reload and resumed training. For ensemble evaluation, invert any transformed member predictions to physical discharge via `transform.inverse` before averaging across members and computing exact NPE.

`CamelsInputs.filled` carries the placeholder positions into hybrid batches. The hybrid encoder sees the stored constant there, marks it as not observed, and never uses it as a reconstruction target.

## Layout

A prepared directory holds `inputs.npy` `[basin, day, feature]`, `attributes.npy` `[basin, attribute]`, their `_was_filled` flags, physical `forcing.npy` `[day, basin, P/T/PET]`, and optional `discharge.npy`, `target.npy` and `target_eligible.npy` `[day, basin]`. `prepared.json` is written last. It records the format version, basin order, dates, field order, fitted transforms with units and fitted counts, the fill value, and the caller's recipe and provenance. Existing output directories are rejected.
