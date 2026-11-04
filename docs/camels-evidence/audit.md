# Independent CAMELS source audit

Read-only audit on 2026-11-04. No checkouts or production edits made. Source archive retained in place; no downloads/extraction. Scripts executed with repository `uv run python` (area parquet inspection via `uv run --group experiments python`). Experiment README, nonlinear-response.md, preparation code and existing test anchors inspected first.

## Sources and identities

- Raw directory: `$HYDROLOGEEZ/data/camels-us/raw/`.
- Archive: `basin_timeseries_v1p2_metForcing_obsFlow.zip`.
- All seven semicolon attribute tables agree on exactly 671 original gauge_id strings. `camels_topo.txt` supplies gauge_lat.
- Forcing member: `basin_dataset_public_v1p2/basin_mean_forcing/daymet/{HUC}/{ID}_lump_cida_forcing_leap.txt`.
- 677 standard filenames: 671 targets plus ['01150900', '02081113', '03448942', '06775500', '06846500', '09535100']. Three irregular members: `03/208111310_forcing.tx`, `06/344894205_forcing.tx`, `02/011230*`.
- Observations: `basin_dataset_public_v1p2/usgs_streamflow/{HUC}/{ID}_streamflow_qc.txt`; all 671 present. Extras ['06775500', '06846500', '09535100']. A nonstandard/archive entry must not define membership.
- Forcing header line 3 is **m²**, not km². Header line 1 differs from attribute gauge_lat. Source readme claims hour=0 but actual forcing labels are hour12: retain dates without UTC shifts.

## Full-record results

All 671 forcing records are consecutive Gregorian 1980-01-01 through 2014-12-31, 12,784 days each; all nine leap days retained. No nonfinite P/T, negative precipitation, temperature sentinel <=-999, or date gaps found. No scientific repair needed for these required variables. This audit does not adopt validation thresholds for unused radiation/SWE/vapor-pressure variables.

Observations contain 8,415,489 source rows. All series are internally daily and have no internal absent dates. 588 start 1980-01-01; 640 end 2014-12-31. Others have shorter leading/trailing coverage, accounting for 162,575 absent dates against full forcing extent. There are 69,481 published missing `-999 M` rows in 424 basins. No nonfinite discharge and no other negative discharge encountered. Flags: A=7,691,797; A:e=654,076; M=69,481; A:<=135. The last flag is real despite omission from source readme; do not reject or filter it. No source observations fall outside full forcing extent; reader selections must nevertheless not truncate global storage.

Absent-date example: 01195100 starts 1981-11-04; 1980-01-01 is absent, distinct from its published 2013-10-01 `-999 M` row. For 01013500, 2014-10-01 is a published missing row (not absent).

## USGS area metadata

Packaged source: `/Users/nicolaslazaro/Desktop/work/RivRetrieve/src/rivretrieve/_internal/providers/usgs_nwis/catalogue/station_metadata.parquet`.
Public API `rivretrieve.metadata(selection, view="source")` returns packaged data offline. The long table has station_id, source_field, source_value, source_unit, state, attribute_role and provenance support fields. Select `drain_area_va` and `contrib_drain_area_va`, units `sq mi`. **source_value is JSON scalar text containing an original string**, so `json.loads` then blank-check before numeric conversion. All 1342 records misleadingly have `state=value`, including 476 blank contributing-area strings. There are 671 positive total areas and 195 positive contributing areas. Never translate blank string to zero. Exact conversion sq mi -> m² is 1609.344². Independent check confirms 15 totals differ from forcing header >10%; 13 of these match contributing area within 10%.

Retain original USGS fields/units separately; do not use them to replace selected Daymet header area. Metadata belongs in preparation, not hot read path.

## Independent numerical examples

Source rows and scalar independently evaluated FAO56/Oudin results saved in `independent_examples.json`. The scalar arithmetic uses only math/date, not production/experiment oudin. Exact cfs conversion: Q * 0.3048³ * 86400 * 1000 / area_m².

Gauge 01013500: header area 2260093113 m²; attribute gauge_lat=47.23739 versus header latitude46.84.

|Date|P mm/day|Tmean °C|Oudin PET mm/day|Q cfs|Q mm/day|
|---|---:|---:|---:|---:|---:|
|1980-02-29|0.01|-17.92|0|285|0.3085156210913192|
|1980-07-01|3.01|14.555|3.3157159628066526|491|0.5315128770380271|
|2014-10-01|0|8.12|1.1928668323644855|-999 M|missing|

PET established convention: attribute gauge_lat; temperature (Tmax+Tmin)/2; FAO56 Eqs21–25 extraterrestrial radiation; lambda2.45 MJ/kg; solar constant0.0820; denominator365 including Gregorian DOY366; floor T+5 at0 only after missing-input validation. Existing tests already provide equatorial equinox and FAO56 Example8 approximate independent anchors, so do not duplicate a broad astronomy test suite.

## Evidence files

The audit scripts and intermediate JSON below remain session artifacts; only
`independent_examples.json` is included beside this report. The portable
`usgs-areas.json` retains original selected area strings and converted values.

- `audit_source.py` and `audit_source.json`: independent basin-by-basin ranges, counts, missing examples and headers (not derived from implementation).
- `area_audit.py` and `usgs_area_rows.json`: original selected packaged USGS fields.
- `independent_examples.json`: precise source lines plus worked values.

No model constants changed or assumptions of research basin eligibility introduced. Count audit considers authoritative attribute-table membership, not filename counts. No checkout created.
