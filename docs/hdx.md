# HDX I/O

HDX is a basin-first scalar Parquet layout for hydrological datasets. Each basin
stores dynamic scalar time series in `basin=<id>/scalar_dynamic.parquet`; basin
attributes live in the root `scalar_static.parquet`; dataset metadata lives in
`manifest.json`. The format is deliberately role-opaque: column names are stored,
but the format does not decide whether a column is forcing, target, prediction, or
model-specific metadata.

hydrologeez owns that semantic layer through `Vocabulary`. By default, `precip`,
`pet`, and `temp` are forcing fields and `streamflow` is the target field. Foreign
datasets can be mapped into these canonical names with overrides:

```python
from hydrologeez.hdx import Vocabulary

vocabulary = Vocabulary({"P": "precip", "E": "pet", "Q": "streamflow"})
```

## Optional install

HDX support is optional:

```bash
pip install 'hydrologeez[hdx]'
```

The numerical model kernel stays format-agnostic and does not import `polars`.
The HDX loader and writer load their Parquet dependencies only when the I/O APIs
are called.

## Loading HDX

Use the top-level API:

```python
from hydrologeez import from_hdx
from hydrologeez.models.gr6j import GR6JForcing

data = from_hdx("path/to/hdx", forcing_type=GR6JForcing)
```

`from_hdx` returns `HDXData(forcing, streamflow, statics, basin_ids, times, mask)`.
For a single basin, arrays are one-dimensional `[T]` and `mask` is `None`. For
multiple basins, dynamic arrays are padded to `[B, Tmax]` and `mask` is a boolean
array with `True` for real records. `times` remains per basin: a single
`datetime64[us]` array for one basin, or a tuple of arrays for multiple basins.

Pass `forcing_type=None` for target-only workflows. In that mode, or when a
required forcing column is absent, `forcing` is `None`; `streamflow`, statics,
basin IDs, times, and mask are still loaded when present.

## Writing predictions

`to_hdx` writes hydrologeez streamflow predictions as a conformant HDX 0.2 scalar
dataset:

```python
from hydrologeez import to_hdx

to_hdx(root, streamflow, times, ["0001"])
```

The writer creates:

- one `basin=<id>/scalar_dynamic.parquet` file per basin;
- root `scalar_static.parquet`;
- a six-field manifest: `format_version`, `name`, `created_at`,
  `producer_version`, `crs`, and `cadence`.

`basin_id` is written as a string and `time` is written as sorted
`datetime64[us]`. The default dynamic field is `streamflow`; pass `field=` when a
different prediction column name is needed.
