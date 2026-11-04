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
