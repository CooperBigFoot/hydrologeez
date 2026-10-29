# Daymet archive and CONUS forcings

Run from the repository root with `uv`. This is a single-writer batch workflow,
not a library API. It does not provision infrastructure or delete the VM.

## Setup and run

```sh
uv sync --group acquisition
uv run --group acquisition python -m scripts.daymet_inventory /data/daymet-inventory
curl --fail --output /data/cb_2024_us_state_500k.zip https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_state_500k.zip
uv run --group acquisition python -m scripts.daymet \
  --inventory /data/daymet-inventory/inventory.json \
  --provider-metadata /data/daymet-inventory/cmr-page-*.json \
  --boundary /data/cb_2024_us_state_500k.zip \
  --stage /data/daymet-stage
```

Discovery records every current annual North America prcp/tmin/tmax file from
NASA collection C2532426483-ORNL_CLOUD (Daymet V4 R1,
https://doi.org/10.3334/ORNLDAAC/2129), preserving original CMR response bytes.
It rejects missing annual triples or gaps since 1980. Use a fresh directory for
each snapshot; review the inventory, storage estimate and throughput before bulk.

Supply `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` in the process environment
or a private mode600 `.env`. Use standard boto3 credentials, either environment
variables or an operator-provisioned private AWS profile. Never put credentials
in arguments or logs. The destination is fixed to `hydrologeez-daymet` at
`https://fsn1.your-objectstorage.com`, region `fsn1`; no other bucket is used.
Earthdata authorization follows only the NASA source/login host allowlist and
is removed on CDN redirects. CLI failures redact exception details because
HTTP errors can contain signed URLs. Remove transient secret copies after use.

Run the command inside an operator-managed persistent service/session on the VM,
not dependent on SSH remaining connected. Run **only one writer** per stage and
bucket prefix. Monitor progress, disk, elapsed time and VM cost externally.
The script does not enforce a financial budget or turn off billing.

## Data and scientific conventions

Raw objects under `raw/Daymet_Daily_V4R1/` are unchanged provider NetCDF files,
including all North America. Provider SHA256 and byte size are checked before
upload. The v4 filenames alone do not establish V4 R1 identity; the collection,
source URLs and CMR revisions in the archived inventory do.

Prepared objects are monthly lossless-zlib NetCDF4, with separate float32 grids
`P`, `T`, `PET` in `(time,y,x)` order. There is no lossy quantization; derived
float64 arithmetic is stored as float32. Chunks are at most `(10,300,300)`.
P is unchanged precipitation (mm/day), T=(tmin+tmax)/2 (degrees Celsius).
PET is mm/day, calculated at each cell before any aggregation:

`PET = Re / 2.45 * max(T + 5, 0) / 100`.

Oudin et al. (2005), [Eq3](https://doi.org/10.1016/j.jhydrol.2004.08.026),
uses latent heat 2.45 MJ/kg. Water density1000 kg/m³ and conversion1000 mm/m
cancel; there is no extra factor1000. Daily temperatures follow this workflow's
purpose, not the paper's long-term climatological fitting experiment.
Extraterrestrial radiation Re (MJ/m²/day) uses supplementary
[FAO56 Eqs21–25](https://www.fao.org/4/x0490e/x0490e07.htm), solar constant
0.0820 MJ/m²/min. This is not a claim of exact Morton AppendixC reproduction.
Astronomy uses actual decoded day-of-year with the FAO365 denominator even
in leap years. Valid T<=-5 gives zero PET. Missing T or latitude stays missing.
P missing does not erase independently valid T/PET. Fill is -9999; valid zeros
remain distinguishable. Outside-CONUS fill has a separate static `conus_mask`.

Native Gregorian dates, original time bounds and noon local-day labels are
retained: February29 exists and December31 is omitted in leap years, giving365
dates per year. These are not UTC accumulation windows or a `noleap` calendar.
No filling, date shifting, regridding or basin averaging occurs.

CONUS is the union of Census2024 1:500,000 cartographic boundaries for the
contiguous48 states and DC. Explicit STATEFP membership lives in
`scripts/daymet_mask.py`. The actual boundary .prj is transformed to the source
CF projection; cell centres covered by the union are included, including the
boundary itself. Simplification and centre selection can exclude coastal fringe
and small islands. Political water areas do not override source missing values.
Native latitude/longitude and projected metre coordinates are retained.

For the inspected 2024 source grid, the computed mask contains7,194,137 centres
in y[4038:6770], x[2610:6989], shape2732×4379. These measurements used exact
inspected endpoint/spacing metadata; each production year recalculates its mask.
Each output records its index window and static mask; the full native mask is
also archived. Source coordinates, projection, units, daily bounds and dates
must agree across all three inputs or preparation stops.

## Bounded storage and verification

Download one annual source trio with at most three independent Earthdata
sessions, one per variable. All downloads join and pass provider checksums before
serial archive uploads and preparation begin. On failure pending work is cancelled
and running transfers finish or fail before the error returns; no completion is
claimed. This does not resume partial byte ranges.

Stage one annual source trio (~30GB, largest observed file14.07GB) and one
prepared month. The inspected crop gives a monthly uncompressed upper bound
4.45GB for P/T/PET plus coordinates. Preparation uses 10×300×300 tiles, tens of
MB per numerical array, a63MB full-grid boolean mask and bounded HDF5 caches;
it does not load annual grids. Allow at least50GB free staging disk.

The 1980–2025 snapshot has138 files,1.2605TB raw. Prepared uncompressed grids
would total2.41TB plus repeated coordinates/masks. A winter central-US 10-day
sample with actual source latitude compressed10.8MB to3.53MB, but it is **not an annual storage forecast**.
Measure actual representative output before relying on compression savings.

`prepared/<processing-hash>/` binds the inventory, boundary, exact scripts,
`uv.lock` and `pyproject.toml`; dependency files are archived with provenance.
It contains `provenance/`, `YEAR/MM.nc`, monthly receipts, annual receipts and a
final `complete.json`. Original CMR bytes, boundary ZIP, scripts and checksums
are archived. Output namespaces change with processing inputs/code.

Every upload is followed by a full streaming S3 GET and SHA256 comparison,
not an ETag or user-metadata assertion. Existing objects are read and compared;
discrepancies fail rather than overwrite. Receipts are published only after
content verification. Source staging is reclaimed only after all12 verified
monthly outputs and the annual receipt. Successful metadata access alone can
never complete a year. Final receipts enumerate bytes, SHA256 and object keys.

Restart the identical command. Complete years/months are reused only after
exact expected object coverage and full content readback; raw receipt hashes
must also match the provider inventory. Incomplete downloads/calculations restart from scratch
via `.part` files; no uncertain byte-range append. Interrupted multipart uploads
may leave uncommitted parts requiring operator cleanup; never delete committed
objects as a retry strategy. Network failures stop the command; rerunning is the
retry mechanism. Keep the same inventory and code to reuse receipts.

## Validation

```sh
uv run --group acquisition pytest
uv run --group acquisition ruff check .
uv run --group acquisition ruff format --check .
uv run --group acquisition ty check
```

Tests use real tiny NetCDF payloads and moto's S3 utility: provider hashes,
stream interruption/retry, prepared-month interruption/restart, checksum GET,
corruption rejection, leap February, missing-vs-zero and independent numerical
radiation/PET values. They do not establish full-history delivery. Bulk completion
requires final bucket receipts/readback, final spatial/date inventory and actual
elapsed/storage/cost reporting. Request approval before VM deletion.
