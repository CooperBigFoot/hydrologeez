# HDX I/O

HDX is a basin-first scalar Parquet layout for hydrological datasets. Each basin
stores dynamic scalar time series in `basin=<id>/scalar_dynamic.parquet`; basin
attributes live in root `scalar_static.parquet`; metadata lives in
`manifest.json`. The format is role-opaque: it stores names without deciding
whether a field is forcing, target, prediction, or model metadata.

**HDX is defined and validated in the canonical
[`hdx` repository](https://github.com/CooperBigFoot/hdx)**; its
[`spec/HDX_SPEC.md`](https://github.com/CooperBigFoot/hdx/blob/main/spec/HDX_SPEC.md)
is normative. hydrologeez implements I/O for that format. The governing rule is:

> **HDX describes the *shape* of data, never *what was done to it*.**

hydrologeez owns the semantic vocabulary. Foreign fields map to canonical names:

```python
from hydrologeez.hdx import Vocabulary

vocabulary = Vocabulary({"P": "precip", "E": "pet", "Q": "streamflow"})
```

## On-disk layout

```text
<hdx-dataset>/
  manifest.json
  scalar_static.parquet
  outlines.geoparquet
  basin=<id>/
    scalar_dynamic.parquet
    gridded_static/<grid-label>.tif
    gridded_dynamic/<grid-label>.zarr
```

For format version `"0.2"`, `outlines.geoparquet` is optional. hydrologeez reads
and writes the scalar manifest, root statics, and per-basin dynamic members.

## Optional install

```bash
uv add "hydrologeez[hdx]"
```

The model core remains format-agnostic. Parquet dependencies load lazily only
when an I/O API is called.

## Loading and conversion

Loading and model conversion are deliberately separate:

```python
import torch

from hydrologeez import from_hdx
from hydrologeez.models.gr6j import GR6JForcing

data = from_hdx("path/to/hdx", forcing_type=GR6JForcing)
torch_data = data.torch(dtype=torch.float64, device="cpu")
```

`from_hdx` returns `HDXData` with a canonical-name forcing dictionary,
streamflow, statics, and mask as NumPy arrays. Times also remain NumPy. A single
basin has dynamic leaves `[T]`; multiple basins are padded `[B, Tmax]` with a
boolean mask. Because model forcing requires `[B, T]`, multi-basin Torch data can
be passed directly, while single-basin NumPy data needs a leading dimension
before model execution.

`.torch()` requires both `dtype=` and `device=` and makes no implicit choice. It
converts forcing, target, statics, and mask, constructs the requested forcing
dataclass, and produces a `torch.bool` mask. Basin IDs and NumPy datetime metadata
are retained. `forcing_type=None`, or missing required fields, produces
`forcing=None` before and after conversion; targets and other available data are
still loaded.

## Writing predictions

`to_hdx` writes a conformant HDX 0.2 scalar prediction dataset:

```python
import numpy as np

from hydrologeez import to_hdx

times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[us]")
streamflow = np.array([1.2, 1.5], dtype=np.float64)
to_hdx("predictions", streamflow, times, ["0001"])
```

The writer creates one dynamic file per basin, root `scalar_static.parquet`, and
a manifest containing `format_version`, `name`, `created_at`, `producer_version`,
`crs`, and `cadence`. Basin IDs are strings and times are sorted
`datetime64[us]`. The default field is `streamflow`; use `field=` for another
prediction name.
