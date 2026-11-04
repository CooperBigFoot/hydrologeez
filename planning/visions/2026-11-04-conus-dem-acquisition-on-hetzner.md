# CONUS DEM acquisition on Hetzner

Collect the current USGS 3DEP 1 arc-second (~30 m) bare-earth elevation tiles
for CONUS once, and deliver the unchanged provider files to
`s3://hydrologeez-daymet/dem/3dep-1arcsec/`. This adds the DEM alongside
existing Daymet and ongoing USGS streamflow acquisition for HBV research.
It is not an ongoing synchronization service or terrain-processing project.

Discussion: [#109](https://github.com/CooperBigFoot/hydrologeez/issues/109).
Successful verified acquisition completes that issue; publishing this vision does not.
[Streamflow #119](https://github.com/CooperBigFoot/hydrologeez/issues/119)
records the existing infrastructure and shared-resource context.

## Delivery

- Use current source releases, not every superseded tile revision. Preserve native
  GeoTIFFs, associated provider metadata/sidecars, and provenance identifying the
  selected releases. Do not convert, mosaic, clip, or regrid them.
- Cover CONUS, not just CAMELS basins. Establish the selected tile inventory and
  record coverage limitations rather than silently omitting unavailable tiles.
- Transfer into object storage through the retained VM without keeping a full
  local DEM copy. Bounded temporary staging is acceptable if needed by the transfer
  mechanism; streaming is preferred. Stay within available disk and memory.
- Use the existing bucket at `https://fsn1.your-objectstorage.com`, with the DEM
  isolated under the prefix above. Do not modify Daymet, streamflow, or the
  separate Pourpoint bucket. Do not provision another VM or bucket.
- Keep acquisition scripts in `scripts/`, not `src/`, and use the repository's
  uv workflow, typed Python, Ruff, ty, and behavior-focused tests.

The infrastructure handoff in #119 identifies `root@167.233.138.69`, VM
`hydrologeez-daymet`, and checkout `/root/hydrologeez`. Workstation access used
key `nicolas-workstation` and AWS profile `hetzner-pourpoint`. Verify current
access and resource availability before execution; do not assume old temporary
credentials remain valid. Coordinate heavy resource use with ongoing streamflow
work. Preserve unrelated files, logs, receipts, services, and credentials.

## Source facts and checks

Use the [official USGS product collection](https://data.usgs.gov/datacatalog/data/USGS:35f9c4d4-b113-4c8d-8691-47c428c29a5b).
The product is 1 arc-second on a geographic grid, not a fixed 30 m projected
raster. It is not the 1/3 arc-second product or a hydrologically conditioned DEM.
CONUS elevations are in metres, with NAD83 horizontal coordinates and NAVD88
vertical reference; verify these against the selected files' metadata.

Initial metadata checks on 2026-11-04 established anonymous access to the
[USGS current tile inventory](https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Elevation/1/TIFF/current/)
and a sample TIFF. No raster was downloaded during discovery. The National Map
catalogue can return multiple revisions of one tile. Current URLs are mutable;
a current XML metadata file can point to a dated object under a historical path.
Do not treat every historical path as obsolete or select the first catalogue hit
without checking release identity.

Native tiles have overlapping margins: the inspected `n39w105` metadata describes
3612 × 3612 pixels, extending beyond its nominal one-degree cell. Preserve these
native files; resolving overlaps belongs to later preparation. Survey acquisition
dates differ from tile production dates. Retain source dates without presenting
this mosaic as an annual observed time series.

Before bulk execution, test a few representative native tiles locally and a
bounded transfer to the destination. Check actual raster contents and metadata,
transfer integrity, interrupted-transfer recovery, and missing-file reporting.
Then run the one-time collection on the existing VM. Leave inventory selection,
transfer tooling, bounded retries, and restart mechanics to implementation;
do not build a general downloader framework.

## Evidence of completion

The selected current CONUS inventory is reconciled against destination objects,
with source identities, sizes and available integrity evidence retained. Report
missing, failed, or unverifiable objects explicitly; a successful process exit
alone is insufficient. Confirm representative destination rasters can be read
and match their source content. Retain a concise acquisition report and enough
progress evidence to restart an interrupted run without blindly repeating the
whole collection. Do not declare delivery complete with unexplained coverage
or transfer gaps.

No final prepared layout, elevation bands, vegetation zoning, basin aggregation,
model API, or calibration changes are included. Existing GRIT flow direction and
accumulation remain separate; do not acquire them again or add NHDPlus terrain.
