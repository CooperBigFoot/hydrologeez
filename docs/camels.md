# Local CAMELS inputs

`hydrologeez.camels` prepares full daily records once and materializes reusable
HBV selections. Basin IDs come from the seven CAMELS attribute tables, not
archive counts or delineation success. Source files are never modified.

## Prepare

```sh
uv run python -m scripts.camels --raw data/camels-us/raw \
  --usgs-areas docs/camels-evidence/usgs-areas.json --output data/camels-us/prepared
```

The checked-in public USGS snapshot makes this preparation reproducible without
a live service or session-specific path. To supply an updated snapshot, use this structure (areas in km²):

```json
{"source": {"provider": "USGS NWIS", "retrieved": "date", "source_units": "sq mi"},
 "basins": {"01013500": {"total_km2": 2253.28965599232,
                         "contributing_km2": 2253.28965599232}}}
```

Supply all attribute IDs. Retain provenance, original source fields/values and
unit conversion in `source`; it is copied into prepared metadata. Decode source
strings before converting; blank contributing areas must be `null`, never zero
or the total area. The reader needs no USGS connection or RivRetrieve dependency.
The audited snapshot used RivRetrieve's packaged USGS NWIS station metadata:
`drain_area_va` and `contrib_drain_area_va`, square miles × 2.5899881103360003.

Output must not exist. Metadata is written last; an interrupted directory is not
a completed store. Keep raw files and the area snapshot for reproducibility.
Generated stores are not source-controlled. `metadata.json` records IDs, units,
source members, full per-basin date bounds, header areas, separate USGS areas,
and invalid/gap counts. `absent_observation_dates` counts missing rows only
within each observation record's own bounds (zero in this source); the audit
separately counts 162,575 absent dates over the full forcing timeline. Per-basin `.npy` arrays contain dense daily forcing,
original cfs, exact QC strings and source-row presence. Observations have their
own full date range, independent of forcing. Negative discharge is preserved in
cfs but becomes NaN in mm/day. No QC filtering is applied.

PET uses attribute `gauge_lat`, Gregorian day-of-year including leap days, mean
(tmin+tmax)/2, and the existing Oudin2005 Eq3 / FAO56 astronomy. The astronomy
denominator remains 365, latent heat 2.45 MJ/kg and solar constant .0820.
Missing/invalid forcing becomes NaN, is reported, and prevents affected selections.
Discharge uses the forcing-header area in m², not USGS or delineated area.

## Select and reuse

```python
from pathlib import Path
import numpy as np
from hydrologeez.camels import Camels
from hydrologeez.hbv_light import Memory, Parameters, simulate

store = Camels(Path("data/camels-us/prepared"))  # metadata only
selected = store.select(["01022500", "01013500"], "1990-01-01", "2000-01-01")
# Dates are [start, stop); basin order is exactly caller order.
# Four simulation rows reuse two basins without copying forcing per parameter set.
parameters = Parameters(
    np.array([0, 1, 0, 1], dtype=np.int64),
    np.tile([0.0, 2.0, 0.05, 0.1, 1.0, 100.0, 1.0, 0.5], (4, 1)),
    np.tile([1.0, 0.5, 0.1, 0.05], (4, 1)),
    np.full(4, 2.5),
)
initial = Memory(np.tile([0.0, 0.0, 50.0], (4, 1)), np.zeros((4, 2)), np.zeros(8))
result = simulate(selected.forcing, parameters, initial)
# Reuse selected.forcing for subsequent parameter sets; normal HBV validation runs.
observed = selected.discharge_mm_day
published_missing = selected.present & np.isnan(observed)
absent = ~selected.present
full_observations = store.observations("01013500")
```

Parameters above are engineering examples, not calibrated recommendations.
Unknown IDs, duplicates, empty windows and dates outside forcing coverage fail.
Selection copies only requested days from read-only memory maps into basin-major
storage (exposed in HBV day/zone/variable order). It is not a zero-copy promise.
Observations and flags follow the same selected basin/date order. Warm-up,
evaluation masks and basin eligibility are separate research choices.


## Static attributes

Prepare all seven local attribute tables independently of daily records or USGS
area snapshots. The output directory must be new; existing daily stores do not
need rebuilding.

```sh
uv run python -m scripts.camels_attributes --raw data/camels-us/raw \
  --output data/camels-us/attributes
```

```python
from pathlib import Path
import polars as pl
from hydrologeez.camels import Camels, read_attributes

attributes = read_attributes(Path("data/camels-us/attributes"))
# A resident Polars DataFrame: 671 rows, gauge_id plus 59 source-named columns.
print(attributes.table.filter(pl.col("gauge_id") == "01013500").select("gauge_name", "p_mean", "geol_porostiy"))
print(attributes.metadata["fields"]["p_mean"])  # description, unit, source, references

store = Camels(Path("data/camels-us/prepared"))
selected = store.select(["01022500", "01013500"], "1990-01-01", "2000-01-01")
# Reuse resident rows in daily selection order, without reading storage:
aligned = pl.DataFrame({"gauge_id": selected.ids}).join(
    attributes.table, on="gauge_id", how="left", maintain_order="left"
)
assert tuple(aligned["gauge_id"]) == selected.ids
# Or materialize only those rows through the reader:
aligned_attributes = read_attributes(Path("data/camels-us/attributes"), selected.ids)
```

`attributes.json` stores typed columns, string identities, and field metadata.
The reader loads this small table only, not raw sources or daily records. It
returns rows in the requested order and rejects unknown, duplicate, or empty
selections. Without IDs it returns all basins in sorted gauge-ID order.

`read_attributes` now returns `CamelsAttributes`, replacing the newly introduced
pandas return value. Use `.table` for the Polars DataFrame and `.metadata` instead
of pandas `.attrs`; `gauge_id` is an explicit column, not an index. Existing JSON
stores remain readable without rebuilding. There is no pandas compatibility API.

Numeric columns use Polars `Float64`; string columns use `String`.
Source `NA` becomes JSON null and Polars null, including missing geological
categories. Zero remains zero. Gauge IDs and `huc_02` retain leading zeros;
names and categories retain their exact source text, including land-cover
leading spaces. No scaling, imputation, category encoding, or clipping occurs.

Field metadata is transcribed from the local `camels_attributes_v2.0.xlsx`
(also supplied as PDF), with descriptions, units, data sources, references,
and section notes about time periods. `documented_name` records documentation
aliases while columns keep raw names, including `geol_porostiy` and
`soil_depth_statsgo`. A blank documented unit stays blank (`slope_fdc`);
`-` stays `-`. Permeability is log10(m²). The documentation labels
`zero_q_freq` as `%` although source values range from 0 to about 0.965;
neither values nor the documented unit are reinterpreted.

The attribute tests compare every local source field and value when the raw
tables are available, and always test missing values, categorical text,
metadata, caller ordering, and alignment with daily selections on small inputs.

## Validation and bounded local evidence

The [independent audit](camels-evidence/audit.md) covers all 671 identities:
1980-01-01 through 2014-12-31, 12,784 Gregorian days, zero forcing gaps or invalid
P/T. Observations contain 8,415,489 rows, 69,481 published `-999 M` rows, and
162,575 absent dates relative to forcing coverage. QC is retained including
`A:<`. No observed records lie outside the full forcing range in this archive;
synthetic tests cover that supported case. All 671 USGS totals are positive;
195 contributing areas are positive and 476 are null.

[Independent source anchors](camels-evidence/independent_examples.json) verify
leap dates, P/T/PET and cfs conversion. Synthetic tests cover ordering, partial
windows, rejection, gaps, QC and resident-input HBV equivalence. Native-data
checks reproduce the audit counts and anchors, and compare discharge plus every
final-memory array exactly against direct resident forcing.

Reproduce selection/runtime checks with
`uv run python -m scripts.camels_measure --store data/camels-us/prepared --evidence measurements.json`.
Time preparation separately; the measurement command does not rebuild the store.

[Measurements](camels-evidence/measurements.json): one local macOS ARM worker,
NumPy 2.5.4, Numba 0.68.1, warm filesystem/Numba caches; not cold-disk or remote
claims. Full preparation: 17.07 s. Metadata open: 3.49 ms. A two-basin, 29-day
selection took 42.88 ms and materialized 1,392 forcing bytes, not the full
205,873,536-byte forcing record. A 128-basin, 3,652-day selection took 677.95 ms
and materialized 11,218,944 forcing bytes. Four repeated evaluations of four
parameter copies took 0.19–5.62 ms for the small case and 198.72–202.59 ms for
the larger one, including normal validation. Per-file mmap opening and first
observed selection costs are included; these are bounded measurements, not a
speed guarantee. Observation arrays add selection memory beyond forcing bytes.
Unlike the earlier forcing-only single-array experiment, this selection opens
four array families per basin and aligns observations. Repeated 128-basin
selections measured 0.47–2.03 s in a follow-up probe; many mmap file opens are a
known cost, not a claim to match the experiment's ~6 ms forcing-only load. The
selection is intended to be reused, not reopened for each parameter evaluation.

Run `uv run --all-groups pytest`, `uv run ruff check .`,
`uv run ruff format --check .` and `uv run --all-groups ty check`.

Implements the [CAMELS input vision](../planning/visions/2026-11-04-camels-hbv-input-preparation-and-readers.md).
Discussion #90 stays open; output storage and later grid-to-basin inputs are separate.


## Hybrid meteorology

Prepare the seven source fields and baseline mean temperature/Oudin PET for
Daymet, Maurer and NLDAS independently of the existing daily and attribute stores:

```sh
uv run python -m scripts.camels_meteorology --raw data/camels-us/raw \
  --output data/camels-us/meteorology
```

The output must be new. `meteorology.json` is the completion marker. No download,
normalization, imputation or feature-set selection occurs. Existing traditional
`Camels` readers are unchanged; observation arrays and static tables are reused,
not copied into a second source archive.

Fields are `dayl`, `prcp`, `srad`, `swe`, `tmax`, `tmin`, `vp`, `tmean`, `pet`,
qualified by product, for example `maurer.prcp`. Metadata retains units, source
member, exact header, source hours, date bounds and derivations. Source values
are float32 (normal rounding from source decimals). The raw archive remains the
exact source. Preparation verifies zero SWE and identical temperatures per basin:
zero SWE is a metadata constant, identical Tmin aliases Tmax, and identical
mean temperature aliases Tmax too. These have no duplicate array payloads;
all original names remain selectable. Other values are retained, including
sentinels. Each selection has separate `valid` and source-row `present` masks.

The local archive's 671 attribute basins have Daymet/NLDAS dates 1980–2014,
but **Maurer ends in 2008**. A combined selection preserves its requested dates;
unavailable product rows are missing, not truncated or filled. A window outside
the combined selected-product coverage is rejected. Physical forcing must be
valid throughout the loaded pool; select an appropriate window/product rather
than silently repairing it.

The [source audit](camels-evidence/hybrid-source-audit.json) found no gaps or
sentinels within these records. All three SWE columns are zero. Maurer and NLDAS
Tmin equal Tmax on every row: matching field names do not establish independent
minimum/maximum measurements. The baseline mean is still `(tmin+tmax)/2`, not a
new scientific interpretation. All source Hr values are 12 despite the readme's
claim of zero; daily dates are preserved without inferring a timezone. Four
Maurer files omit date names from their header but contain the same 11 data
columns. No separate thermal/longwave radiation field is available.

### Select once, reuse resident batches

```python
from pathlib import Path
import torch
from hydrologeez.camels import Camels, read_attributes
from hydrologeez.camels_meteorology import CamelsMeteorology
from hydrologeez.hybrid.camels import load_inputs
from hydrologeez.hybrid.data import Normalization
from hydrologeez.hybrid.model import ModelConfig

features = ("daymet.prcp", "maurer.prcp", "nldas.prcp")
config = ModelConfig(features=features, attributes=("gauge_lat",))
inputs = load_inputs(
    CamelsMeteorology(Path("data/camels-us/meteorology")),
    ["01013500", "01022500"],
    "1990-01-01",
    "2000-01-01",
    features=features,
    physical=("daymet.prcp", "daymet.tmean", "daymet.pet"),
    attributes=read_attributes(Path("data/camels-us/attributes")),
    attribute_names=config.attributes,
    observations=Camels(Path("data/camels-us/prepared")),
)
batch = inputs.batch(config, context=slice(0, 365), simulation=slice(0, 365))
# Only fit on a deliberately authorized training/preprocessing pool:
normalization = Normalization.fit(batch.meteorology, batch.padding, features)
# Exclude warm-up or evaluation dates explicitly, without restoring missing targets:
valid = torch.ones((365, 2), dtype=torch.bool)
valid[:30] = False
batch = inputs.batch(config, context=slice(0, 365), simulation=slice(0, 365), evaluation_mask=valid)
```

The example's latitude is an unscaled engineering input, not a recommended
scientific attribute set. Select and scale attributes explicitly; numeric finite
columns are accepted, categorical encodings and missing-attribute policies are
caller-owned. All source static attributes remain accessible through
`read_attributes`. Discharge-derived attributes must not enter a supposedly
unlabeled pool accidentally.

Omit `observations` for pretraining/inference pools that must not read discharge.
Meteorology-only `CamelsMeteorology.select` needs neither a physical product nor
PyTorch; it can supply authorized preprocessing or pretraining pools even where
no complete physical forcing is available. The hybrid convenience adapter
requires valid physical forcing for its reusable runtime batch. It never fits
normalization. Genuine missing neural values become NaN; raw values and source
masks remain in `inputs.source`. Synthetic hiding is supplied separately to the
runtime, and these dense windows have no padding. Observations retain original
cfs, QC, source presence and Daymet-header-area mm/day conversion; no QC filtering
is imposed. Target validity excludes nonfinite/missing observations and can only
be narrowed by `evaluation_mask`. Initial memory can be supplied with
`dataclasses.replace(batch, initial=memory)`.

`batch` verifies ordered feature/attribute names against `ModelConfig`. Context
and simulation slices may differ. Slice batches share resident CPU storage;
list basin indexing and device transfers copy the selected batch. Do not mutate
shared tensors. Load a selected basin/date pool once, then reuse it rather than
opening files for each training step. Three precipitation estimates are never
automatically averaged. Physical P/T/PET remain separate, unnormalized values.
The same batches support direct discharge, masked pretraining, frozen-encoder
training and inference described in [hybrid HBV](hybrid-hbv.md).

See the [layout decision and measurements](../planning/experimentation/2026-11-10-camels-inputs-for-efficient-hybrid-hbv-training.md)
for loading, copies, memory, conversion and bounded real-data runtime evidence.
No remote VM performance or accuracy improvement is claimed.
