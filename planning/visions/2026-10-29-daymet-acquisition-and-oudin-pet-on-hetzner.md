# Daymet acquisition and Oudin PET on Hetzner

Produce an owned archive of Daymet source files and usable daily precipitation (P), mean temperature (T), and Oudin potential evapotranspiration (PET) grids for the contiguous United States. This supports subsequent HBV calibration and a dataset of calibrated parameters, not a broad meteorological collection.

Context: [discussion #111](https://github.com/CooperBigFoot/hydrologeez/issues/111), under [Program #87](https://github.com/CooperBigFoot/hydrologeez/issues/87). This vision supersedes the discussion's raw-acquisition-only handoff: acquisition and PET preparation are now one end-to-end workflow. Successful delivery of the archive, prepared grids, and verified run completes #111. Publishing this document alone does not complete it.

## Outcome and boundaries

- Use Daymet V4 R1 precipitation (`prcp`), minimum temperature (`tmin`), and maximum temperature (`tmax`). Collect the full available temporal history with no study-period cutoff. The last verified inventory was 1980–2025; recheck NASA's catalog before execution and record the chosen inventory snapshot.
- Preserve original North America annual NetCDF files and provider metadata unchanged in the dedicated bucket. These extend beyond CONUS; do not describe a spatial subset as an unchanged source file.
- Produce separate CONUS daily P, T, and PET grids at Daymet's native nominal 1 km spacing, retaining the native grid/projection rather than regridding. Document the precise CONUS extent/mask used.
- P is the source daily precipitation in mm/day. T is `(tmin + tmax) / 2` in degrees Celsius. Calculate Oudin PET per grid cell from that temperature, latitude and date, in mm/day, before any future basin aggregation. No basin averaging belongs here.
- Respect Daymet's native calendar, including the omission of December 31 in leap years. Verify decoded dates and astronomical day-of-year handling; do not silently fill missing days, shift subsequent dates, or claim UTC-day alignment. Streamflow alignment is later work.
- No NLDAS, Maurer, ERA5-Land, DEM, vegetation acquisition, calibration, or application refactor. Scripts belong in `scripts/`, not `src/`.
- Choose a practical prepared storage format and bounded-memory layout. No Zarr requirement. Broader reader/runtime optimization remains [#90](https://github.com/CooperBigFoot/hydrologeez/issues/90), not a prerequisite to this delivery.

## Scientific reference

Read Oudin et al. (2005), “Which potential evapotranspiration input for a lumped rainfall–runoff model? Part 2—Towards a simple and efficient potential evapotranspiration model for rainfall–runoff modelling”, Journal of Hydrology 303, 290–306, DOI https://doi.org/10.1016/j.jhydrol.2004.08.026.

The verified local PDF is:

```text
/Users/nicolaslazaro/Desktop/resources/items/05fd933d-057f-4a3e-866b-9107ab0d184a/content/Ludovic_Oudin_2005_Which_potential_evapotranspiration_input_for_a_lumped_rainfallrunoff_model_Part_2Towards_a_simple_and_efficient_potential_evapotranspiration_model_for_rainfallrunoff_modelling.pdf
```

The Oudin method is settled; its exact numerical formulation has not yet been verified or implemented. Use the paper, not recollection from training, to establish the equation, coefficients, temperature threshold/cold-temperature rule, radiation and energy-to-water units. Identify and cite the astronomical radiation calculation used, distinguishing the paper's specification from any supplementary reference. Record assumptions explicitly and test representative numerical cases, including cold days, latitude/season variation, missing inputs and leap-year handling. Missing source data must remain distinguishable from valid zero precipitation or zero PET. Escalate a material scientific ambiguity rather than silently choosing a different PET method.

## Source and credentials

- NASA collection: `C2532426483-ORNL_CLOUD`; DOI https://doi.org/10.3334/ORNLDAAC/2129.
- Access guide: https://daymet.ornl.gov/getdata ; catalog discovery via NASA CMR.
- Known HTTPS base: `https://data.ornldaac.earthdata.nasa.gov/protected/daymet/Daymet_Daily_V4R1/data/`.
- Filenames: `daymet_v4_daily_na_{prcp,tmin,tmax}_{year}.nc`. Preserve V4 R1 identity separately: filenames alone say v4.
- Earthdata credentials are already in the ignored repository-root `.env`, with mode 600, under `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`. Authentication is untested. The supplied username is an email address and may need correction if NASA expects a different username.
- Load secrets without displaying them, placing them in command arguments, committing them, or including them in manifests/logs. Establish secure VM credential delivery and cleanup. Do not assume `earthaccess` or dotenv loading is already installed/configured.

## Provisioned infrastructure

Use the resources already created; do not provision a duplicate VM or reuse the Pourpoint bucket.

- Hetzner CLI context/project: `pourpoint`, expressly approved for this dedicated VM.
- VM: `hydrologeez-daymet`, server ID `173357768`, Falkenstein (`fsn1`), CPX32, 4 shared x86 vCPUs, 8 GB RAM, 160 GB local disk, Ubuntu 24.04.
- SSH: `root@167.233.138.69`, existing key `nicolas-workstation`. Key-based login was verified from the user's workstation. Approximately 143 GiB was free at creation.
- Dedicated firewall: `hydrologeez-daymet-fw`, ID `11922652`, inbound TCP 22 only. No application service needs public ingress.
- Dedicated bucket: `s3://hydrologeez-daymet`, endpoint `https://fsn1.your-objectstorage.com`, region `fsn1`. Creation and `head-bucket` succeeded.
- Local AWS CLI profile: `hetzner-pourpoint`. Its name is not the destination bucket. Use the dedicated hydrologeez bucket only; do not modify other project resources or bucket contents.
- The VM was provisioned on 2026-10-29 and is already billing. Quoted compute price at creation: EUR 0.0615089/hour, capped at EUR 38.36469/month including account tax, plus IPv4. The user's USD 50 ceiling applies to VM spending only, not storage. Track cumulative VM cost from provisioning and stop/escalate before exceeding the ceiling; do not assume monthly caps prevent an overrun across months or that powering off ends billing.
- Storage costs are separate. Verify inventory sizes and report the expected raw-plus-prepared footprint and recurring charges before bulk execution. Do not assume the archive fits within the base storage allowance. No hard storage budget was specified.

## Execution

1. Test authenticated access locally with bounded representative transfers. Verify actual original-grid NetCDF contents, units, coordinates, fill values and dates; catalog records and successful login alone do not establish correct payloads.
2. Discover the complete selected inventory, including sizes and available provider integrity metadata. Confirm region/version/variables and exact coverage. Avoid the historical seven-variable size estimate. Measure representative throughput to estimate duration and cost rather than promising a transfer rate.
3. Build and test a restartable acquisition-and-preparation workflow in `scripts/`. Use the repository's uv workflow, typed Python, Ruff, ty and test-first guidance. Add appropriate project dependencies rather than relying on the orchestration kernel's environment.
4. Deploy code, environment and necessary secrets securely to the existing VM. Run the bulk workflow there, independently of a workstation SSH session. Process bounded pieces: the VM cannot stage the entire archive. Upload and verify raw files and derived outputs before reclaiming temporary storage. Test restart/retry behavior locally or with bounded VM work before the full run.
5. Run the complete selected history. Preserve useful progress, failures, provenance and completion records in the dedicated bucket. Avoid silent partial completion and destructive overwrites. Verify object integrity using suitable checksums/metadata; multipart S3 ETags are not universally file MD5 checksums.
6. Report the verified final inventory and residual limitations. Remove transient secret copies and report ongoing VM billing. Ask the user to approve VM deletion once the delivered data and records are safe; deletion is not implicit in publication of this vision.

The implementing agent is expected to execute the bulk run, not stop after writing scripts. Bulk execution follows successful bounded tests and inventory/cost reporting; new material access, scientific, storage or budget blockers must be surfaced. Ordinary implementation choices such as file grouping, concurrency and retry mechanics are delegated, within memory/disk limits and provider access rules.

## Evidence of completion

- A reproducible command and setup/run instructions, with no embedded secrets, plus meaningful test and lint/type-check results.
- Raw source coverage matches the recorded provider inventory for all three variables and all available years. No unexplained missing or corrupt objects; manifests include source URLs, version, coverage and verification evidence.
- Prepared CONUS P/T/PET grids cover that same native daily record with documented coordinates, units, missing-data policy, Oudin references and processing provenance. Verify spatial/date coverage and numerical samples, including PET against independently calculated reference cases.
- Restart/interruption evidence demonstrates that completed work is reused safely and incomplete uploads or calculations are detected.
- Verified reads from the final bucket, not merely successful local computation. Report object prefixes, sizes, period, gaps/source limitations, actual elapsed time and cost estimates, with enough information for later work under #90.
