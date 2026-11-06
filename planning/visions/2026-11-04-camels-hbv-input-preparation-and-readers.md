# CAMELS HBV input preparation and readers

Implement the first local CAMELS-US input path for repeated HBV parameter sweeps. Prepare inputs once, select basins and dates efficiently, and reuse the selected inputs across parameter sets without changing HBV numerics.

Discussion: [#90](https://github.com/CooperBigFoot/hydrologeez/issues/90), within [Program #87](https://github.com/CooperBigFoot/hydrologeez/issues/87). This vision delivers the agreed CAMELS input slice; it does not complete the broader #90 discussion, which also covers later Hetzner/grid-to-basin inputs. Publication alone delivers no implementation.

## Prepared inputs

- Use all **671 CAMELS attribute-table basins**, identified by their original string gauge IDs. Keep research inclusion separate from preparation. Do not select basins by archive filename counts or by successful Pourpoint delineation.
- Retain original source files. Source data are from [Zenodo record 15529996](https://zenodo.org/records/15529996), already locally available under `data/camels-us/raw/`. Use the time-series archive and relevant attribute tables; model-output archives are unnecessary.
- Prepare the **full available daily record**, not an experiment-specific window. Audit actual date coverage and missingness across all basins. Retain Gregorian dates and leap days; do not impose the native-grid Daymet 365-record convention on these CAMELS files.
- Use basin-average CAMELS Daymet precipitation and mean temperature `(tmin + tmax) / 2`, with Oudin PET. This is one zone per basin, without DEM or vegetation zoning.
- Reuse the established Oudin preparation convention rather than introducing a new PET method. The input-path experiment documents the existing implementation, astronomy, latitude choice, units, and independent numerical anchors. Missing forcing must not become plausible PET through clipping. Audit and report forcing gaps or invalid values; do not silently fill, truncate, or drop basins. An affected simulation selection must not run with invalid forcing. If source defects require a scientific repair policy, report them rather than inventing one.
- Retain full daily observed streamflow, original discharge values/units and source QC flags. Align observations with the requested forcing basin/date selection. Preserve missing observations explicitly, including dates absent from the source; never turn sentinels into physical flow, fill observation gaps, or apply additional QC filtering during preparation. Preserve the distinction between absent dates and published missing-value rows where present in source evidence.
- Convert observed discharge to **mm/day using each CAMELS Daymet forcing-header area**, not a newly delineated area. Keep original discharge so this choice remains reversible. Retain USGS total and contributing area metadata separately, without treating blank contributing areas as zero or silently substituting one definition for another.

## Storage and reader behavior

Use **local dense NumPy prepared storage**, retaining explicit basin identities, dates, variables, units and the preparation conventions needed to interpret the data. Consecutive dates for a basin should support efficient slicing; memory-mapped `.npy` storage is the established approach. Exact file grouping, metadata representation and public API are implementation choices, not a requirement to reproduce the experiment's script interface.

Select requested basins and dates with deterministic identity/order semantics. Reject unknown or unsupported selections clearly rather than silently substituting basins, shifting dates, or shortening windows. Prepare/materialize a selection once and reuse it across repeated HBV evaluations, sharing forcings across parameter sets instead of expanding their stored copies. Selection may require a real copy; do not promise zero-copy multi-basin selection. Observations and forcing must refer to the same ordered basin/date selection.

Keep full-record preparation independent of experiment dates, warm-up and evaluation masks. Preserve observations outside a particular forcing selection in the prepared record rather than discarding them globally. Metadata-only or small-window access must not require eagerly loading the full forcing store.

Use the existing HBV input contract and validation. Do not change equations, introduce a validation bypass, or add another runtime engine to implement the reader. Avoid making the benchmark machinery a production dependency. Use the project's `uv`, typed Python, Ruff, ty and test-first conventions.

## Evidence and important distinctions

[Input-path experiments](../../experiments/hbv-input-path/README.md) found dense basin-major NumPy inputs strongest for the tested larger local selections; compressed Zarr remains a later alternative, not an implementation requirement. Read the [current nonlinear-response rerun](../../experiments/hbv-input-path/nonlinear-response.md), not only the historical UZL/K0 timings. These are bounded, warm-cache, single-worker local results, not guarantees for remote storage or the eventual roughly 10,000-parameter research workload.

The experiment audited only 128 basins over its prepared window. It found 677 standard forcing filenames plus irregular archive members, while all seven attribute tables agreed on 671 basin identities. Its source checks and synthetic reordered-ID/date-selection tests are useful evidence, not a substitute for the full-record audit required here.

The [USGS area audit](https://github.com/CooperBigFoot/hydrologeez/issues/90#issuecomment-6634205686) found positive total areas for all 671 in RivRetrieve's packaged metadata, contributing areas for 195, and blank contributing fields for 476. Fifteen total areas differ from the CAMELS header by more than 10%; contributing areas explain agreement within 10% for thirteen of those. Header area is not necessarily the spatial polygon area used to average forcing. The selected header convention is explicit, not a claim that all area definitions agree. RivRetrieve's public metadata API can expose the separate USGS fields; a new live observation acquisition is not required. Keep this metadata concern out of the hot reader path.

The GRIT delineation run processed all 671: 538 succeeded, 133 failed 100 m snapping, and 135 successful results differed from header area by over 10%. These are diagnostic flags, not preparation exclusions. [Pourpoint #223](https://github.com/pourpoint-tools/pourpoint/issues/223) investigates holes separately. Neither the 403 within-10% basins nor any hole-filling rule is an adopted scientific eligibility policy.

## Evidence of completion

- An auditable full-record preparation covers the authoritative 671 identities, reports coverage/forcing problems and observation missingness, and retains the agreed source and area/QC information without silent scientific repair.
- Independently checked source examples prove precipitation, mean temperature, PET and cfs-to-mm/day conversion, including leap dates and missing observations. Existing suitable tests and utilities should be reused, not duplicated.
- Reader tests establish reordered basin selection, partial time windows, correct observation alignment, explicit invalid-selection behavior and reuse across parameter sets. Selected forcing agrees with its independently checked prepared values, and representative HBV runs agree with direct resident-input runs for discharge and final memory.
- A bounded local measurement separates preparation from selection/loading and repeated consumption, demonstrates that a small time selection need not load the full dataset, and reports real materialization costs. No universal speed threshold, extensive benchmark suite or remote-storage claim is required.
- Provide concise usage showing preparation, basin/date selection, and reuse with the current HBV runtime. Run the relevant project tests, Ruff and ty; keep generated stores out of source control.

## Outside this vision

Output/state storage and streaming belong to [#128](https://github.com/CooperBigFoot/hydrologeez/issues/128). Scores, normalization, clustering, calibration algorithms, evaluation periods, warm-up policy and basin eligibility remain experiment concerns. Also excluded: Hetzner layout, raw Daymet acquisition, gridded aggregation/regridding, DEM/vegetation zoning, remote Zarr adoption, Pourpoint fixes, and numerical/runtime redesign.
