# USGS streamflow acquisition on Hetzner

Acquire all available published daily mean discharge for USGS stations in CONUS using RivRetrieve, and deliver the observations and unchanged source quality information to owned Hetzner object storage. This is data acquisition for later HBV calibration, not calibration or quality screening.

Context: [#119](https://github.com/CooperBigFoot/hydrologeez/issues/119), under [Program #87](https://github.com/CooperBigFoot/hydrologeez/issues/87). Successful acquisition and verified delivery complete #119's current streamflow scope. Publication alone does not complete it. Pourpoint and watershed delineation mentioned in the original issue body are explicitly deferred.

## Dataset contract

- Select all USGS stations with published daily mean discharge inside the contiguous 48 states plus DC. Use a documented geographic boundary, not a bounding box alone. Basin suitability, drainage area thresholds and minimum record length are not selection criteria.
- Collect all available years, not only the Daymet period. Establish actual source-accessible history and discovery coverage; do not silently treat catalogue absence, failed retrievals or inaccessible historical records as no data. Record the selected station/series inventory, available coverage, gaps and unresolved limitations.
- Retrieve published daily mean discharge, not instantaneous data reaggregated to daily. In RR this is provider `usgs_nwis`, quantity `discharge`, frequency `daily`, statistic `mean` (USGS parameter `00060`, statistic `00003`). Preserve distinct source-series identities; do not average or arbitrarily select among matching series at one station.
- Retain source daily dates unchanged. The later alignment decision is to match USGS and Daymet by local calendar date, acknowledging uncertain exact interval agreement. Do not label USGS daily dates UTC or apply time shifts. Keep the full USGS calendar, including days absent from Daymet; no temporal join or Daymet-calendar reduction belongs here.
- Preserve source approval status and qualifiers unchanged and available with their observation identities. Their use, interpretation and filtering are out of scope. Preserve missing values and distinguish absent observations from retrieval failures. No QC-based exclusion, imputation or calibration-ready transformation.
- Use RR's native cache format, plus result bundles retaining publisher payloads for source quality fields. No new replacement streamflow storage format is required. Retain metadata and provenance needed to understand selection, source series, acquisition and units.

## Established RivRetrieve facts

The sibling checkout is `../RivRetrieve` (case matters), remote `https://github.com/RivRetrieve/RivRetrieve`. Investigation used commit `9026dd9b41372bf5d3a2b3669b586dba72d3a6ac`; pin and record the actual revision used for acquisition and verify its behavior. Read its [USGS provider documentation](https://github.com/RivRetrieve/RivRetrieve/blob/main/docs/providers/usgs_nwis.md) and [usage documentation](https://github.com/RivRetrieve/RivRetrieve/blob/main/docs/usage.md).

The inspected catalogue contains 26,258 station records, 26,201 selectable USGS stations, and 24,495 daily-mean-discharge stations across 24,630 series. These are national snapshot counts, not the final CONUS selection or guaranteed current coverage. Catalogue checks were dated 2026-09-22; published record start/end fields were all null. Establish history through appropriate source discovery and retrieval rather than treating those nulls as coverage bounds. Check catalogue freshness and report any limitation on exhaustive station discovery.

All inspected station rows declare `EPSG:4269` (NAD83), derived from USGS `dec_coord_datum_cd`. Do not assume or relabel them WGS84. Use compatible CRS handling for the CONUS selection and preserve the declared coordinate provenance.

The provider already uses the modern USGS Water Data API daily collection. Returned daily timestamps display midnight with `time_zone="unknown"`; this is not proof of a midnight UTC window. Published USGS daily values and Daymet's approximate local-day forcing need not have identical physical support. This acquisition makes no exact-alignment or leakage-elimination claim.

### Cache and quality preservation

- `RIVRETRIEVE_CACHE_DIR` selects a local filesystem cache. Live fetch defaults to `cache="bypass"`, which does not cache. Use explicit caching policy. `reuse` reuses covered scopes; `refresh` obtains the current source answer. Successful empty results and failures have different coverage semantics.
- The accumulated cache preserves RR's native-unit observations and source evidence, but currently omits USGS `approval_status` and `qualifier`. Normalized public results also omit them. Do not assume cache acquisition alone preserves quality information.
- On original live acquisition, `receipts=True` returns `publisher_payload` receipts. `rr.to_bundle(result)` preserves exact payload bytes and provenance; `rr.from_bundle` restores them. Keep these bundles durably and document how to access quality fields by source time-series ID and source date. Payloads may contain padding beyond the requested interval.
- Cache-hit receipts can be `store_excerpt`, not original USGS payloads. They cannot recover omitted quality flags. A mixed result can contain both receipt kinds; verify preservation for the actual acquired scope.
- Cache commit precedes caller-side bundle persistence. Treat a safely persisted bundle and its acquisition record as part of completion. If caching succeeded but the bundle is missing, reacquire the affected scope with publisher receipts; the new source version may differ, so retain acquisition provenance rather than claiming recovery of the original vintage.
- Cache writes commit at the end of a public fetch, not each HTTP page/station. Use bounded work and durable progress so interruption does not require restarting the national acquisition. Batching, concurrency and ledger details are delegated.
- Current cache publication relies on local filesystem operations. Do not point it at an S3 URL or assume an object-store mount offers equivalent guarantees. Audit with `rr.audit_cache("usgs_nwis")`; inspect/recover interrupted state through the supported APIs. Publish consistent snapshots, not an actively changing store.

A bounded live test on 2026-10-31 fetched station `07374000` for 2024-01-01–2024-01-02 into an isolated empty cache using `cache="reuse", receipts=True`. It returned two normalized rows and a publisher receipt containing six padded source days. Both requested observations had `approval_status="Approved"` and `qualifier=["ESTIMATED"]`: these are independent source facts. Exact receipt bytes and origin survived bundle roundtrip; subsequent reuse returned store excerpts, and the cache audit passed. This verifies the small workflow, not bulk throughput, full-history discovery, interruption recovery or remote restore.

Optional upstream improvements are [RR #621: quality retention](https://github.com/RivRetrieve/RivRetrieve/issues/621) and [RR #620: remote cache support](https://github.com/RivRetrieve/RivRetrieve/issues/620). Neither is a prerequisite. Do not expand this effort into implementing them.

## Infrastructure and destination

Reuse the existing VM; do not provision a replacement or delete it. The user explicitly asked to keep it alive. Follow the [#119 infrastructure handoff](https://github.com/CooperBigFoot/hydrologeez/issues/119#issuecomment-6547628180):

- VM `hydrologeez-daymet`, ID `173357768`, Hetzner project/context `pourpoint`, fsn1, Ubuntu 24.04, 4 shared vCPUs, 8 GB RAM and 160 GB disk. Last reported free space was about 142 GiB; recheck before use.
- SSH `root@167.233.138.69`, workstation key `nicolas-workstation`. Checkout `/root/hydrologeez`; uv installed.
- Working cache and staging: `/root/streamflow/`. Keep this separate from Daymet work.
- Durable destination: `s3://hydrologeez-daymet/streamflow/usgs/`, endpoint `https://fsn1.your-objectstorage.com`, region `fsn1`. Workstation AWS profile `hetzner-pourpoint`. The profile name is not authorization to use the Pourpoint bucket.
- Agreed logical prefixes: `cache/` for published RR-native snapshots, `receipts/` for result bundles preserving publisher quality fields, and `manifests/` for selection, provenance, progress and verification. Generation naming and consistent publication mechanics are delegated; readers must be able to identify a complete matching delivery.
- Preserve all Daymet objects and receipts, `/root/daymet-inputs`, and `/root/daymet-logs`. #111 bulk execution succeeded, but final verification/cleanup was still being reconciled at handoff. Coordinate before heavy shared-resource work. Do not restart the Daymet service or assume its temporary credentials remain available.
- Verify access without displaying secrets. Do not place credentials in command arguments, commits, manifests or logs. An optional USGS API key can increase allowed request volume; assess current provider limits and respect them.
- VM billing continues, previously estimated around EUR 1.50/day. Report current expected compute/storage costs before bulk execution and track them. No new infrastructure or material cost expansion without user approval; do not inherit the earlier Daymet budget as a fresh allowance.

## Execution and evidence of completion

Implement acquisition scripts in `scripts/`, not application code in `src/`. Use the hydrologeez uv workflow, typed Python, Ruff and ty, and repository test-first guidance. Keep the solution focused on this acquisition; do not refactor RR or hydrologeez unnecessarily.

The implementing agent owns routine operational choices: full-history discovery, boundary source, request sizes, pagination, source rate limits, retries, local disk budgeting, checksums, snapshot transfer and restoration. Investigate these facts rather than asking the user to design the machinery. Surface material scope, access, scientific or cost blockers rather than weakening the dataset contract silently.

Start with bounded representative local/VM work. Demonstrate cache plus quality-bundle preservation, interrupted acquisition/resume, upload and restore/readback before the full run. Measure practical footprint and throughput; the full archive and staged cache replacements must fit the VM or the agent must surface the constraint. Avoid one giant uncheckpointed request and respect current cache concurrency guarantees.

Then execute the full acquisition and delivery, not just author scripts. Completion requires:

- Reproducible commands, pinned dependency/source revisions, selection boundary/CRS provenance and an explicit CONUS station/series inventory.
- All accessible daily mean discharge history for that inventory, with successful empty answers, missing observations and failed scopes distinguished. Resolve retryable failures; disclose any remaining source-access limitations and do not claim unexplained partial results are complete.
- Native cache snapshots, publisher-receipt bundles and matching manifests in the agreed bucket prefixes. Quality fields must remain available without reacquiring USGS data; document their retrieval from bundles. No quality filtering.
- Verified local cache integrity, source quality preservation and restart behavior. Record relevant tests, lint and type-check results.
- Verification from the delivered object-storage copy, including restoration and usable RR reads, with receipt quality fields still accessible. Successful local computation or upload acknowledgements alone are insufficient.
- Final station/series counts, actual time coverage, object inventory/bytes, integrity evidence, residual limitations, elapsed time and cost estimates, plus the canonical completed delivery location.

Leave the VM alive, preserve the Daymet work, and report continuing billing and any remaining transient files or credential cleanup. No model calibration, basin delineation, quality-screening policy, UTC reaggregation or Daymet matching is part of delivery.
