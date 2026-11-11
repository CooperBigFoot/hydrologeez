# Regional model benchmark setup and GPU smoke tests

Prepare a reproducible experiment comparing conditional flows, alpha/delta HBV and a pure deep-learning LSTM for spatial and temporal regionalization. Deliver working data preparation and training/inference scripts, realistic smoke-test evidence, a reproducible blocker report, and a recommendation for assigning independent training jobs to GPUs. Stop for human approval before the full campaign. The purpose is to find promising approaches, not optimize every method or establish compute-efficiency claims.

This is a standalone setup vision, not a reproduction of Feng et al. (2023). It builds on the completed foundations in [Program #170](https://github.com/CooperBigFoot/hydrologeez/issues/170); publication or completion does not reopen or complete that Program. No full scientific training campaign, time-varying parameters, dropout ensembles, static compression, or additional ablation tournament is authorized.

## Models and outputs

There are 21 variants per regional fold:

- 16 performance-conditioned MAF configurations: approximately 50k, 100k, 300k and 1M trainable weights crossed with 100, 1,000, 10,000 and 100,000 examples per training basin. Keep one architecture family and report actual counts. Context is 45 static predictors plus achieved calibration-period NPE. Train one flow per configuration/fold; sample ten static parameter vectors per basin/request at inference.
- Four hybrids: alpha-HBV and delta-HBV, each with LSTM or KDA. Delta jointly trains encoder and parameter head against discharge. Alpha first reconstructs masked meteorology, then freezes the encoder and trains the parameter head through HBV.
- One standalone LSTM predicting discharge directly, the established pure-DL comparator to beat. It is not an HBV encoder or an exact reproduction of the paper's inputs/loss.

Use one recurrent layer, width 256 for all comparators. Hybrid parameter head: 256 -> 256 with SiLU -> 13 bounded HBV parameters. Alpha reconstruction head: 256 -> 256 with SiLU -> 16 meteorological outputs. KDA uses the existing single-head architecture. Standalone LSTM: 61 input values -> LSTM-256 -> linear scalar discharge readout. Enforce an explicit physically valid output policy if needed; report it rather than silently clipping predictions. Train ten independent models per comparator/fold and average their discharge predictions equally.

Reference counts with current input wiring: standalone 326,913; LSTM hybrid encoder+parameter head 428,557; KDA hybrid 385,643; alpha pretext head adds 69,904. Hybrid encoders currently receive 16 values plus visibility/masking indicators (48 channels) and 45 statics. Counts are implementation checks, not equal-compute claims. Alpha/delta within each encoder family must share inference architecture and capacity.

All HBV parameters are static through the simulation, using the exact GA search bounds in [HBV-light](../../docs/hbv-light.md), not current hybrid default ranges. Physical Daymet P, mean temperature and Oudin PET must match the GA inputs. HBV laws, discharge units, target masks and evaluation handling must be consistent between flow and hybrid predictions.

Average member discharge at each timestep, then compute exact NPE on the ensemble-mean hydrograph. Never average parameters or member scores as a substitute. Archive individual scores and parameter vectors too. No selection of ensemble members using test discharge.

## Regions, dates and input boundaries

Use seven folds: hold out one whole region and train on the other six. HUC02 groups are R1=[01,02], R2=[03,06], R3=[04,05,07], R4=[09,10], R5=[08,11,12,13], R6=[14,15,16,18], R7=[17]. Exact mapping is verified against Feng Figure B1 and the authors' Zenodo PUR scripts. Source pool is 671 basins. Author-aligned headline evaluation uses the archived 531-basin subset; retain outputs for all eligible held-out basins so broader summaries remain possible. Verify target coverage and report failures/missing observations, not silently substitute basins.

- Supervised training/GA-label period: 1989-10-01 through 1999-09-30.
- Primary test: held-out region, 1999-10-01 through 2005-09-30.
- Secondary saved result: held-out region during the supervised decade; no extra plot required.
- Alpha pretraining: all regions' meteorology from 1980-01-01 through 1999-09-30, no later weather. This is intentional transductive geographic access, not held-out discharge access.
- Existing GA simulation starts 1987-10-01, with two years of warm-up before scoring. Recommend and document consistent test warm-up/state handling without fitting on test discharge.

Hold out whole validation basins inside the six training regions. Exclude their discharge-derived flow examples from training as well. Fit shared static scaling only on training basins. Keep alpha meteorological-pretraining normalization within its authorized historical pool; supervised comparators must not fit normalization on test-region observations. No discharge signatures, basin IDs or HUC codes as predictors. Declare the historical context used to estimate each hybrid's static parameters; do not silently re-estimate parameters over test windows or let later-period weather enter training. Normal runtime consumption of test forcing is necessary for simulation, not permission for test-time training.

Validation selects checkpoints, flow configuration and requested score for headline comparisons. Test-region discharge is unavailable to training, tuning and selection. Final test inference occurs only after those decisions are fixed. Requested performance is not a guarantee, particularly after temporal transfer.

## Exact feature lists

All models receive the same 45 static predictors, without compression. Eighteen landscape fields:

`elev_mean`, `slope_mean`, `area_gages2`, `frac_forest`, `lai_max`, `lai_diff`, `gvf_max`, `gvf_diff`, `soil_depth_pelletier`, `soil_depth_statsgo`, `soil_porosity`, `soil_conductivity`, `max_water_content`, `sand_frac`, `silt_frac`, `clay_frac`, `carbonate_rocks_frac`, `geol_permeability`.

For each of `daymet`, `maurer`, `nldas`, add product-qualified `p_mean`, `pet_mean`, `aridity`, `p_seasonality`, `frac_snow`, `high_prec_freq`, `high_prec_dur`, `low_prec_freq`, `low_prec_dur`. These are already computed from 1980-01-01 through 1999-09-30 with Oudin PET. Aridity is mean PET / mean precipitation, not a discharge-derived index. Original published CAMELS climate fields are preserved but not selected. Published landscape descriptors remain fixed descriptors, not a reconstruction of historically available mapping.

The hybrids and standalone LSTM receive 16 separate daily channels:

- Daymet: `prcp`, `tmin`, `tmax`, `srad`, `vp`, `dayl`.
- Maurer: `prcp`, `tmean`, `srad`, `vp`, `dayl`.
- NLDAS: `prcp`, `tmean`, `srad`, `vp`, `dayl`.

Maurer/NLDAS Tmin and Tmax are identical in these archives; include temperature once. SWE is all zero and excluded. No observed or lagged discharge input. The flow has no daily-weather encoder. HBV separately receives unnormalized `daymet.prcp`, `daymet.tmean`, `daymet.pet`, with mean=(Tmin+Tmax)/2 and Oudin PET.

Alpha masks 75% of days per context window, hiding all 16 channels together before encoding. Statics remain visible; reconstruction MSE scores only hidden valid values. This borrows timestamp masking from Ti-MAE, not its Transformer architecture or a claim that 75% is optimal. No masking-ratio study now.

## Training and persistence

Use discharge MSE for hybrids and standalone LSTM, conditional negative log likelihood for flows, masked reconstruction MSE for alpha pretraining; exact NPE is the reporting metric. Soft NPE was considered and superseded because of memory cost. State the MSE reduction/normalization precisely and do not silently replace it with another objective.

Train each flow, delta model and standalone LSTM for 50 epochs; alpha gets 50 pretraining plus 50 discharge epochs. One epoch is one full pass through the defined training dataset. For flows that is the selected rows; for sequence models define a finite window dataset and explain length, stride, overlap and warm-up, rather than inventing arbitrary random-batch epochs. Recommend these sequence settings from hydrological needs and measured feasibility.

Target physical batches of 1,024 (2,048 may be tested) and investigate gradient accumulation. Measure actual throughput and peak memory; smaller physical microbatches are allowed in smoke tests and must be recommended explicitly if large batches do not fit. Accumulation changes effective batch size, not physical capacity. Do not change sequence objective or drop basins merely to satisfy a batch target.

After every effective batch/optimizer update, atomically refresh a resumable latest checkpoint, not an indefinitely growing archive of every update. Include optimizer, scheduler, RNG, data position and configuration needed for verified restart. Retain best-validation and final models; preserve learning curves and reasons for any failed run. Validation inference during training is allowed. Full test inference and plots are deferred until training/selection finish. Save sufficient metadata for inference in a separate process after the training GPUs are gone.

The full future campaign comprises 112 flow fits and 350 comparator fits, with alpha pretraining as an additional stage. Do not launch it under this vision.

## Local evidence and data preparation

Local workspace root: `/Users/nicolaslazaro/Desktop/uzh-results`. These files/data are not bundled in GitHub; check availability before remote staging.

- `2026-11-10-flow-training-examples/HANDOFF.md`, `retrieved/full/manifest.json` and archives: 67,771,000 scored vectors, 671 basins x ten independent GA seeds, 7.06 GB retrieved and verified. Raw archives remain immutable. They are optimization trajectories, not posterior draws.
- `2026-11-11-flow-subset-audit/REPORT.md`, `summary.json`, `per_basin.jsonl`: all basins have 100,495–100,526 unique valid strictly interior float64 vectors; no original bound endpoints, invalid scores or conflicting duplicate scores. Nested 100/1k/10k/100k sizes are feasible. Deduplicate at stored precision, verify again after training-dtype conversion, and report rather than silently clip/jitter endpoints. Design nested sampling covering independent runs and score ranges. The 0.50–0.80 range is ONLY the requested-performance plot grid, never a training filter. Six basins' maximum archived NPE is below 0.50.
- `2026-11-11-regional-split/feng2023-pur-findings.md`, `feng2023-pur-candidate-folds.json`, 671/531 CSVs: verified grouping and author evaluation IDs. Archive DOI 10.5281/zenodo.7091334 resolves to 7943626. Archived scripts contain different example dates/FDC settings; reuse split evidence, not those training recipes.
- `2026-11-11-historical-climate-attributes/README.md`, `REPORT.md`: reproduction, tests and hashes. `data/camels-us/attributes/attributes.json` contains original 59 columns plus 27 new columns; `selected45.json` and `climate-provenance.json` identify experiment inputs. All 18,117 new values finite; originals preserved. Snow uses author-code T<=0 convention; seasonal harmonic uses fixed sample means and zero-based day/365.25. No discharge read for derivation.
- `data/camels-us/meteorology`, `data/camels-us/prepared`, `data/camels-us/raw` hold prepared meteorology, discharge and source records. Verify coverage, masking, area conversion and input identity before upload. Stage only required files, with checksums and a frozen manifest.
- `~/Desktop/flow-scaling-mock/` contains synthetic PNG/SVG plotting mocks and scripts, never scientific results.

## Existing core and missing functionality

Reuse [CAMELS readers](../../docs/camels.md), [hybrid HBV/runtime](../../docs/hybrid-hbv.md), `hydrologeez.flow` and [shared losses](../../docs/losses.md). Source inspection found an LSTM encoder but no standalone discharge LSTM model/trainer/checkpoint interface. Do not assume that comparator already exists. Build the smallest reusable missing core functionality in hydrologeez; keep campaign-specific assembly, sampling and plotting in uzh-results. Avoid a second general training framework. Update hybrid bounds to the experiment's GA contract explicitly, without silently changing unrelated users' defaults.

Current KDA is a plain single-head PyTorch recurrence, not optimized CUDA kernels; differentiable HBV is also recurrent. Tiny CPU/MPS checks do not establish GPU throughput, realistic gradient memory or training stability. Width 256 with batch 1,024 implies very large KDA state history. Treat CUDA execution, maximum feasible sequence/batch and complete training-stage transitions as evidence questions.

Core defects found during smoke tests must be reported with reproduction and proposed fixes. Fix experiment plumbing within scope; do not silently alter scientific requirements or adopt surrogate physics to bypass core problems. Material core defects require a clear human-facing repair proposal; a blocked setup is an honest outcome, not permission to claim launch readiness.

## Required smoke tests and GPU recommendation

Use project-native uv, typed Python, Ruff, ty and focused test-first development. CPU/local checks precede paid GPU tests. Cover every distinct path: flow likelihood and conditional sampling; delta LSTM/KDA backward through HBV; alpha reconstruction then frozen-encoder discharge training for both encoders; standalone LSTM. Check finite meaningful gradients, intended parameter updates/freeze, bounds, feature order, no target leakage, ten-member inference/averaging, exact NPE, physical parity where relevant, checkpoint reload and restart (including accumulated-update boundaries). Verify actual 45/16 input integration, not only synthetic dimensions.

Run bounded realistic GPU smoke tests with representative real basin/time windows, not merely tiny toy cases. Measure memory, examples/second, step/checkpoint time, numerical failures and projected full-run duration. These are operational feasibility measurements, not a compute-efficiency scientific study. Exercise largest flow and recurrent workloads, input transfer and saved-model inference. Preserve commands, environment/hardware identity, logs and failures.

Use the repository's Runpod skills and installed CLI. This vision authorizes bounded smoke-test provisioning only, not a full training launch. The user's total credit ceiling is $170 for setup, future campaign, storage and inference combined, not $170 for smoke tests alone. Before provisioning, provide a concrete bounded smoke-test spend/runtime cap and reserve; obtain agreement if feasibility requires a material portion of the credit. Verify live price/stock, CUDA compatibility, persistent storage, spend guards and cleanup. Do not leave paid GPUs idle during development or downloads.

Recommend assigning independent fold/configuration/seed jobs to independent GPUs, not distributed training of one model by default. Specify which family runs on which GPU type, jobs per GPU, microbatch/accumulation choices, storage layout, observed constraints, expected campaign cost and uncertainty. More GPUs need not be better value; never assert the full 462-fit campaign fits $170 without measurements. Stop for approval of the organization and any necessary compromises.

Keep output footprint measurable. Persist model checkpoints, individual/ensemble discharge, parameter vectors, scores and provenance, not every recurrent activation. Estimate final bytes from real smoke artifacts before launch. Download reasonably sized results, verify checksums, then remove compute; if output approaches/exceeds 200 GB, propose a retained-storage/selective-download solution rather than automatically transferring or deleting it. No destruction of the only verified copy.

## Future result contract and completion

Implement reproducible final analysis for:

1. 4x4 heatmap of median basin NPE for flow variants.
2. One CDF panel: validation-selected best flow, four hybrids, pure LSTM. Solid ensemble-mean-discharge score curve, thin transparent member curves of matching colour; legend gives median basin NPE in parentheses.
3. Requested-versus-achieved plot for all 16 flows: requests 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80; median across basins of ensemble-mean-discharge NPE, diagonal reference. Colour=model size and line style=data size. Individual parameter-score conditioning need not yield an ensemble exactly on the diagonal.

No titles or explanatory prose inside figures beyond axes/ticks and agreed legends. No extra plots required. Smoke outputs must be clearly distinguished from campaign results.

Completion means the agent delivers a reproducible setup, smoke evidence, explicit failures/blockers and a GPU organization recommendation, with documented remaining human decisions. Recommend unresolved engineering settings (windows, validation manifest, exact flow architecture, nested sampling, optimizer/schedule, requested-score selection and state handling) rather than quietly treating them as previously agreed. Do not claim scientific performance, launch the full campaign, or adopt a workaround merely because the setup vision is complete.
