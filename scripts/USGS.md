# USGS daily discharge acquisition

Run from the repository with `uv run --group acquisition python -m scripts.usgs`.
RR is pinned in `uv.lock`; acquisition uses public daily mean discharge, unchanged
source calendar dates, all matching series, and no quality filtering.

## Bounded pilot

```sh
uv run --group acquisition python -m scripts.usgs acquire-scope /root/streamflow/pilot --station 435444087462300 --start 2006-10-31 --end 2006-11-02
uv run --group acquisition python -m scripts.usgs verify /root/streamflow/pilot
```

Each scope refreshes RR's local native cache, saves an original publisher result
bundle, then commits its marker. A missing or changed bundle forces reacquisition,
not cache reuse: recovered source vintage may differ. Failed result bundles remain
for diagnosis. A process lock serializes acquisition, validation and publication.
Use `recover WORKSPACE` for RR-supported interrupted-cache recovery. Do not alter
native manifests or integrity seals. Completion requires the saved acquisition
outcome IDs to remain in the current cache metadata. Use nonoverlapping pilot
scopes in a workspace: overlapping refreshes can supersede earlier vintages and
require reacquisition. Generated acquisition-plan scopes do not overlap.

## Inventory and bounded acquisition

Download the Census 2024 1:500k states boundary from
https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_state_500k.zip.
The inventory retains its checksum, CRS and the contiguous 48 states plus DC.

```sh
uv run --group acquisition python -m scripts.usgs inventory WORKSPACE BOUNDARY.zip --max-history-probes 10
uv run --group acquisition python -m scripts.usgs plan WORKSPACE --max-days 100000
uv run --group acquisition python -m scripts.usgs acquire-plan WORKSPACE --max-scopes 1
```

Inventory unions live USGS Daily/Mean metadata with RR's retained identities.
Coordinates retain declared CRS and are transformed for polygon membership.
Unknown bounds are not empty results: paired earliest/latest source queries resolve
a bounded number of series per invocation. Explicit successful empty source answers
remain distinct from failures. Resume retains the same discovery snapshot; use a
fresh workspace for new discovery. Inspect saved rate-limit/Retry-After headers
before resuming a failed request. There is no automatic request retry loop here.

Planning refuses unresolved geography, history or catalogue access. `max-days` is
an operator-selected sizing limit, not a scientific coverage cutoff; measure small
and full-history scopes before choosing it. `acquire-plan` resumes at most the given
number of unfinished scopes. No minimum record length, Daymet cutoff or basin filter
is applied. Metadata coverage does not promise continuous observations; bundles retain
missing-value rows, absent dates, outcomes and source series for later gap analysis.

## Delivery and verification

```sh
uv run --group acquisition python -m scripts.usgs publish WORKSPACE UNIQUE_GENERATION --profile hetzner-pourpoint
uv run --group acquisition python -m scripts.usgs restore-files EMPTY_WORKSPACE UNIQUE_GENERATION --profile hetzner-pourpoint
uv run --group acquisition python -m scripts.usgs verify-remote EMPTY_WORKSPACE UNIQUE_GENERATION --profile hetzner-pourpoint
```

Destination is fixed to `s3://hydrologeez-daymet/streamflow/usgs/` at the Hetzner fsn1
endpoint. Credentials come from the AWS profile/environment, never arguments or
manifests. Publication holds the workspace lock, verifies uploaded bytes by full
GET SHA256, and writes the matching generation marker last. Completed generations
are immutable. A delivery marker certifies matching bytes, **not national acquisition
completion**. Inspect inventory unresolved rows and `planned_scopes_complete`.

**Native local cache copies are not portable at the pinned RR revision**: inode and
device integrity witnesses change on download ([RR #651](https://github.com/RivRetrieve/RivRetrieve/issues/651)).
`restore-files` verifies downloaded bytes and publisher bundles, not native local
cache usability. Do not run local `verify` or resume acquisition on that copied cache.
`verify-remote` instead uses RR's supported public S3 reader to audit the delivered
native snapshot and compare cached reads with restored result bundles. Keep original
local working cache for acquisition; remote snapshots are read-only deliveries.

Verification rejects publisher fallback receipts, but RR `reuse` can attempt source
requests for an incomplete cache. Independent offline verification can deny source
HTTP in a separate test process; production code does not patch transport.

## Source quality

```python
from pathlib import Path
import rivretrieve as rr
from scripts.usgs_acquire import quality_rows

result = rr.from_bundle(Path("receipts/RESULT.rrbundle").read_bytes())
for properties in quality_rows(result):
    print(
        properties["time_series_id"], properties["time"], properties.get("approval_status"), properties.get("qualifier")
    )
```

These are unchanged publisher properties. Payloads can include padded dates outside
the requested scope. Join by source time-series ID and source date, not station/date
alone. Approval and qualifiers are independent source facts; neither is screened.

Implementation context: [vision](../planning/visions/2026-10-31-usgs-streamflow-acquisition-on-hetzner.md)
and [#119](https://github.com/CooperBigFoot/hydrologeez/issues/119). Scripts and pilot
publication alone do not complete full acquisition.
