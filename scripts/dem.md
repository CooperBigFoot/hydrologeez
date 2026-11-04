# Native CONUS 3DEP acquisition

One-time collection for [the published vision](../planning/visions/2026-11-04-conus-dem-acquisition-on-hetzner.md).
No raster conversion or clipping. The source is the official USGS current **1 arc-second** collection.

```sh
uv run --group acquisition python -m scripts.dem_inventory --boundary cb_2024_us_state_500k.zip dem-inventory
uv run --group acquisition python -m scripts.dem_acquire --inventory dem-inventory/inventory.json --stage dem-stage
```

Obtain the Census 2024 500k state ZIP from the URL recorded in `scripts/daymet_mask.py`.
Discovery preserves the boundary and paginated provider listings beside the inventory.
Selection intersects nominal one-degree cells with the 48 states and DC, including islands
present in that generalized boundary. Native overlapping margins stay unchanged.
Every listed `USGS_1_<tile>.*` file is retained, including auxiliary sidecars.
Other basename families are excluded: the current tree includes stray `USGS_13` files
(the wrong resolution) and Windows `Thumbs.db` caches.
Absent TIFF/XML cells are reported, not silently dropped or classified as land gaps:
coastal ocean cells and generalized-boundary differences need investigation against provider coverage.
An unresolved missing cell prevents a complete report.

The inventory pins each current object by ETag, size and modification date. Conditional GET
rejects a changed current object. Original XML retains its publication and survey dates and may
link to the selected dated release under `historical/`; that link does not imply an obsolete release.
XML fields are preserved even when their reference system, units or resolution are unexpected;
field-specific discrepancies appear in `metadata_warnings`. Invalid XML is also retained with
a parsing warning, not accepted as verified spatial metadata. The tile/product title and dated
release link are checked separately. Citation publication dates need not equal filename dates.
Dated official S3 links for the selected tile are checked against the frozen TIFF size and ETag.
Alternate provider URLs without comparable identity evidence, unavailable historical objects,
and identity differences are recorded in `unverifiable_releases`; they do not prevent preserving
the frozen current bytes.
These limitations do not prove that the current raster is wrong or that its datum is verified.
Auxiliary `.tif.aux.xml` files are retained without treating them as FGDC metadata.

Transfer is sequential with one temporary source file (largest selected object plus an 8 MiB
reserve required). Source reads use 8 MiB blocks; destination upload uses the existing bounded
archive transfer (two workers, 64 MiB multipart chunks). Three source-request attempts are allowed.
Single-part provider ETags are checked as MD5; available provider checksum headers are retained
without claiming CRC validation. Every destination object is read back in full and compared by
SHA256 and size. Multipart ETags are identity validators, not whole-file checksums.

Only `s3://hydrologeez-daymet/dem/3dep-1arcsec/` is written. Configure AWS credentials/profile
through the standard boto3 environment; the endpoint is `https://fsn1.your-objectstorage.com`.
Keep the discovery directory and stage on persistent VM storage. Per-file receipts are stored
under the inventory SHA256. Restart the same command: receipted objects are read back and
verified without source download. Existing release-checked receipts remain immutable and their
recorded limitations are included in the new report. Use a fresh stage directory when recovering
a run whose local report must be preserved; destination reports are content-addressed.
An upload interrupted before its receipt is safely compared
with the source on restart. Corrupt destination objects are reported, never silently overwritten.
Unfinished multipart uploads may require operator inspection after an abrupt process kill.

`dem-stage/report.json` and content-addressed destination reports list verified objects,
failed files, missing cells, metadata warnings and release limitations. `transfer_complete` and
exit success mean all selected bytes reconciled with no missing cells. The conservative `complete`
field also requires no metadata or release limitations; it is not an operator completion decision.
Assess documented limitations against the acquisition requirements. Neither flag replaces the required independent coverage review,
representative destination raster reads or the acquisition completion report.
Before bulk execution, coordinate VM resources with streamflow and run a bounded representative
source/destination check. Neither tests nor these commands authorize another bucket or VM.

Tests: `uv run --group acquisition --with responses pytest tests/test_dem.py`.
