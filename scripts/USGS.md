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
uv run --group acquisition python -m scripts.usgs inventory WORKSPACE BOUNDARY.zip --refine-locations --max-history-probes 10
uv run --group acquisition python -m scripts.usgs plan WORKSPACE --max-days 100000
uv run --group acquisition python -m scripts.usgs acquire-plan WORKSPACE --max-scopes 1
```

Inventory unions live USGS Daily/Mean metadata with RR's retained identities.
Coordinates retain declared CRS and are transformed for polygon membership.
`--refine-locations` reconciles the frozen inventory with complete Canadian
monitoring-location pages and missing-coordinate station lookups. Recorded foreign
countries veto polygon membership; country/polygon conflicts remain visible.
Missing country is not interpreted as US. Source US sites without coordinates stay
unresolved, except station `11516100`: the user-approved polygon-selection exception
requires USGS monitoring-location country `US` and state `41` (Oregon). Its selection
is included, while coordinates remain missing and `polygon_selection` remains
`unknown_coordinates`. The inventory retains `selection_exception`, the authoritative
monitoring-location feature and response provenance. Re-run `--refine-locations` on
an unplanned inventory to apply this exception using retained source evidence.
Station `11341370` has no geographic exception: successful empty earliest/latest
source probes describe current source availability, not a location or proof that no
historical records ever existed; its geography remains unknown. Unknown-coordinate
series are also eligible for paired history probes. Confirmed `source_empty` rows
stay in `series` and `availability_limitations` with their source evidence, but do
not block planning or create acquisition scopes. This is a source-access limitation
of the discovery snapshot, not a geographic exclusion. Unknown, failed or nonempty
history with unknown geography still blocks planning. Successful pages checkpoint
before continuing, and errors stop the invocation for an explicit retry. Refinement is refused after planning.
Unknown bounds are not empty results: paired earliest/latest source queries resolve
a bounded number of series per invocation. Explicit successful empty source answers
remain distinct from failures. Resume retains the same discovery snapshot; use a
fresh workspace for new discovery. Inspect saved rate-limit/Retry-After headers
before resuming a failed request. There is no automatic request retry loop here.
Set `USGS_API_KEY` in the process environment to authenticate inventory and RR
observation requests. Inventory sends it only as `X-Api-Key` to HTTPS
`api.waterdata.usgs.gov`, never in URLs or saved request provenance; redirects
are refused to avoid forwarding credentials. The CLI does not load `.env` files.
History probe pairs start at least 10 seconds apart when the key is nonempty,
or 180 seconds apart without it. Every invocation starts with the same cooldown.
Elapsed request/checkpoint time counts; slow requests do not trigger catch-up
bursts. These fixed intervals target 720 requests/hour against the published
keyed 1000/hour allowance, or 40 against the unkeyed 50/hour/IP allowance.
Key presence is not proof of validity or available quota. Other requests sharing
the key/IP consume that allowance; pacing only covers history pairs, not metadata
pagination or RR observation acquisition. A failed pair or incomplete location refinement
stops history probing. The CLI summary exposes `history_failures` and
`location_refinement_status`; inspect these before continuing. Completed pairs
checkpoint immediately. Cancellation during a wait preserves them; an interrupted
pair is probed again on explicit resume. Respect saved `Retry-After` before resume;
the initial cooldown does not override a longer source-requested wait.

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

`restore-files` verifies downloaded bytes, audits the restored native cache and
compares cached reads with publisher bundles. The pinned RR revision verifies
copied cache content and renews local filesystem integrity witnesses through its
public APIs ([RR #651](https://github.com/RivRetrieve/RivRetrieve/issues/651)).
No manual seal changes are needed. `verify` can check the restored local workspace;
`verify-remote` checks the published snapshot using RR's public S3 reader. Remote
snapshots remain read-only deliveries.

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
