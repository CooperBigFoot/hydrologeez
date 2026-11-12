# Reduced regional flow and LSTM campaign

Run the reduced regionalization experiment through verified results, not another open-ended feasibility study. Compare performance-conditioned flows, alpha-HBV–LSTM, delta-HBV–LSTM and a pure discharge LSTM on unseen regions in later years. Implement missing transformed-output/loss integration, persist preprocessing, validate locally, stage inputs to Runpod, smoke-test on GPU, then execute the campaign if correctness and measured budget checks pass. Infer, plot, retrieve and verify artifacts before cleanup.

This vision implements [discussion #196](https://github.com/CooperBigFoot/hydrologeez/issues/196). Complete it only after the reduced campaign and verified result handoff; publication or a blocked run is not completion. Return findings and blockers there. Related core work is [#192](https://github.com/CooperBigFoot/hydrologeez/issues/192), which is not complete merely because generic preprocessing exists; close it only when its remaining output/loss contracts are delivered and verified.

This supersedes the broader campaign scope and approval-to-launch stop in the [original setup](2026-11-11-regional-model-benchmark-setup-and-gpu-smoke-tests.md) and [feasibility follow-up](2026-11-11-regional-benchmark-learning-and-gpu-feasibility-follow-up.md). Those documents remain evidence, not requirements to run KDA or sixteen flows. This vision authorizes the reduced full campaign after the gates below. Stop for human decisions on material blockers, required scientific changes or insufficient budget; do not silently shrink epochs, ensembles, data or folds.

## Fixed scientific scope

Seven variants per fold:

- Four float64 MAFs: approximately 100k and 1M weights crossed with 1,000 and 10,000 unique examples per training basin. Existing four-transform, three-layer conditioner widths 92 and 332 yield 100,200 and 998,760 weights; reuse and verify them. One trained flow per configuration/fold, ten parameter samples per basin/request. Context is 45 statics plus NPE.
- Alpha-HBV–LSTM, delta-HBV–LSTM and pure discharge LSTM: one recurrent layer, hidden width256, ten independent seeds each fold. Hybrid head 256 ->256 SiLU ->13 static HBV parameters; alpha reconstruction head 256 ->256 SiLU ->16. No separate unfrozen static branch, no static compression, no dynamic HBV parameters.

There are 28 flow fits, 210 supervised comparator fits and ten shared alpha pretraining runs. KDA, extra flow tiers, dropout ensembles and new architecture/feature tournaments are excluded.

Use seven verified HUC02 groups: [01,02], [03,06], [04,05,07], [09,10], [08,11,12,13], [14,15,16,18], [17]. Hold out each entire region in turn. Source pool671; author-aligned headline evaluation531, retaining outputs for eligible full-pool held-out basins too. Preserve the existing stratified15% whole-basin validation manifest inside each fold's six training regions; exclude validation basins from discharge supervision and flow rows.

Supervised scoring: 1989-10-01 through1999-09-30. Primary held-out-region test: 1999-10-01 through2005-09-30. Retain secondary spatial-only results on1989–1999. Alpha pretraining uses all regions' historical weather/statics, 1980-01-01 through1999-09-30, without discharge or later weather. These geographic transductive inputs are intentional and disclosed.

Physical HBV uses untransformed Daymet precipitation, mean temperature and Oudin PET, matching the original GA campaign. Use exact GA HBV-light bounds, common units and observation masks. Parameters stay fixed for each basin/member simulation. Estimate hybrid parameters from historical context only, preserving the existing once-per-basin1989–1999 context protocol; primary physical simulation starts1997-10-01, scoring starts1999-10-01. Secondary simulation starts1987-10-01, scoring1989-10-01. Compute warm-up from actual calendar dates (including leap days), not hardcoded730.

## Inputs and no additional QC

Reuse the exact selected45.json list: eighteen landscape attributes (`elev_mean`, `slope_mean`, `area_gages2`, `frac_forest`, `lai_max`, `lai_diff`, `gvf_max`, `gvf_diff`, `soil_depth_pelletier`, `soil_depth_statsgo`, `soil_porosity`, `soil_conductivity`, `max_water_content`, `sand_frac`, `silt_frac`, `clay_frac`, `carbonate_rocks_frac`, `geol_permeability`) plus nine climate fields per Daymet/Maurer/NLDAS (`p_mean`, `pet_mean`, `aridity`, `p_seasonality`, `frac_snow`, `high_prec_freq`, `high_prec_dur`, `low_prec_freq`, `low_prec_dur`). Product-qualified climate fields are already computed on1980–September1999 using Oudin PET. Do not substitute original longer-period climate summaries or discharge signatures.

Sequence models receive sixteen separate channels: Daymet `prcp,tmin,tmax,srad,vp,dayl`; Maurer and NLDAS each `prcp,tmean,srad,vp,dayl`. Flows receive no dynamic weather inputs. No observed/lagged discharge input, basin ID or HUC predictor.

**Do not run or apply additional GSIM QC, statistical outlier, constant-run or homogeneity screening in this experiment.** This is the human's explicit scope reduction, preserving GA calibration-label consistency. Existing QC utilities/archive may remain untouched but must not silently change eligibility. Preserve original source missingness and invalid-target handling, provider flags as evidence without new exclusions, and reject missing/invalid observations from transform fitting, losses and metrics. Validate dates, IDs, units, finite physical forcing and source alignment. No additional QC does not mean no data validation. If new incompatible masks are necessary, stop and explain implications for archived GA labels instead of silently rescoring or reusing inconsistent labels.

## Persisted preprocessing and reusable core integration

PRs #194/#195 delivered persisted preprocessing and QC, not the whole of #192. Inspect current code and [experiment preprocessing docs](../../docs/experiment-preprocessing.md). Reuse existing `hydrologeez.preprocessing`, prepared readers, LSTM and hybrid runtimes; do not build another framework. Fill missing reusable output-head, differentiable transformed-loss, inversion and checkpoint contracts in hydrologeez with focused tests and independent review. Campaign orchestration stays in uzh-results. Core implementation changes follow normal review/merge policy; this publication is not a blanket implementation-PR merge authorization.

Fit global per-feature z-scores, not per-basin normalization:

- Shared fold training-basin static scaler for flow, delta and pure LSTM; each basin counted once. Fit delta/pure-LSTM meteorological channel scalers on the fold's authorized historical training inputs, explicitly recording dates including any warm-up/context used. Reuse across seeds.
- Alpha weather/static preprocessing is fitted once on the explicitly authorized all-region1980–September1999 unlabeled pool. Apply this SAME artifact during pretraining, frozen-head training and inference across every fold. Do not feed fold-rescaled statics into a supposedly shared frozen encoder.
- Every fold has its own discharge transform, fitted on accepted supervised-training observations only, excluding validation/test basins, warm-up and dates outside the scoring decade. Reuse it across all discharge-trained comparators in that fold, including alpha. Alpha's unlabeled permission never extends to discharge statistics.

Discharge is in mm/day. Training transform is `log1p(Q)` followed by global z-score, equivalent to the generic `Log(offset=1.0, units="mm/day")` then global ZScore. Do not confuse it with GSIM's `log(Q+0.01)` in m³/s. Preserve physical discharge and an explicit eligibility mask alongside transformed targets. Fit before any input filling; no filled target contributes to training or metrics.

Train pure LSTM and hybrids with the same valid-element-weighted MSE in transformed discharge space. HBV receives physical forcings and emits physical discharge; transform predictions differentiably inside its loss. Resolve pure-LSTM output semantics explicitly: standardized log discharge can be negative, so a softplus head directly constrained in that space is invalid. A physically constrained output with transform inside loss or another coherent reversible readout is acceptable; document the choice, units and nonnegativity behavior without silent clipping or double transformation. Store all transforms/readout metadata with saved models and verify independent reload inference. Flow training remains conditional likelihood; archived NPE labels remain unchanged.

Prepared datasets must contain final model-input values and masks, fitted parameters, pools/recipe, basin/date/field order, units and hashes. Loading must not fit or transform again; use identity model-input normalization where the prepared adapter requires it. Keep physical P/T/PET separately. Do not automatically append filled flags as predictors and change the agreed input count; retain mask semantics and explicitly verify handling if missing predictors exist.

## Training, selection and alpha reuse

Train50 epochs; alpha50 reconstruction plus50 frozen-head epochs. Alpha masks75% of days across all sixteen weather channels together, retaining visible statics, and scores reconstruction only on hidden valid values. Ten seed-distinct shared pretraining checkpoints feed fold-specific supervised heads; ensure heads are initialized/restarted independently as intended and never carry supervised weights between folds. Cache frozen encoder outputs only for identical deterministic input windows/normalization/weights; prove equivalence, preserving day/window identities.

Start recurrent physical/effective batch256 with no accumulation and flow batch2048. Use existing Adam1e-3/cosine-to5e-5 schedule as the initial campaign setting, documenting readout/gradient-clipping details. Keep corrected finite sequence windows (nominal1095 days,365 warm-up,730 scored,stride365), explicit supervised date masks and actual calendar warm-up calculation. Verify all scored elements lie in the training decade, overlap/tail coverage, partial final batches, optimizer steps per epoch and valid-element loss reduction. One epoch is a complete pass over defined windows or selected flow rows.

Deduplicate GA rows at float64 precision; preserve strict interior bounds and nested1k/10k subsets with existing reproducible seed/generation coverage. Audit sampling rather than claim round-robin alone guarantees score balance. Never restrict training scores to the plot interval.

Select checkpoints and headline flow configuration/request using validation only. Keep final and best-validation models. Freeze and record selection rules before test inference: transformed-MSE validation for supervised comparators, likelihood checkpoint criterion for flows plus the existing median validation-basin ten-member ensemble NPE rule for configuration/request selection are starting policies; disclose their distinction. Use the existing validation request grid .50–.90 step.05, while the steering figure remains .50–.80. Do not claim likelihood guarantees discharge improvement. Report any scientifically consequential selection change before using it.

Run all ten members in physical discharge space. Invert transformed outputs before averaging timestep-wise, then compute exact NPE on ensemble-mean physical discharge. Archive member scores too; never average scores/parameters or pick members using test discharge. Final primary/secondary inference follows training and validation selection, not per-epoch test peeking.

## Checks before paid campaign execution

Reuse existing evidence but recognize raw-MSE pilots do not validate log-MSE training. Test transform fit isolation, round-trip/readout behavior, physical-vs-transformed loss, gradients, frozen alpha preprocessing reuse, no double normalization, mask/date alignment, checkpoint metadata and recovery. Compare transformed losses with independently computed examples; demonstrate physical-unit ensemble semantics.

Run local end-to-end checks first, then bounded real-data GPU smoke/learning pilots covering flow, delta-LSTM, alpha pretraining/frozen training and pure LSTM under the new preprocessing. Inspect training/validation curves and predictions, not just finite outputs. Do not demand a particular score or guaranteed improvement. Confirm reasonable learning opportunity for the50-epoch schedule; stop and report collapsed, unstable or clearly inadequate training rather than spending the whole budget. No new hyperparameter tournament.

Latest checkpoints must resume models, optimizers/schedulers, RNG, data position and loss accumulators correctly. Reuse verified date/restart fixes. Start flow durable cadence500 updates; recurrent latest at completed optimizer updates where overhead is reasonable. Measure real durable I/O and bounded work loss. Local `/tmp` is not the sole recovery copy. Preserve best/final weights plus required inference metadata; retain optimizer state until training/restart needs are finished. Validate separate-process recovery before launch.

## Runpod execution, budget and artifacts

Total project ceiling remains **$170**, not account balance. Reconcile prior receipts (latest reported cumulative spend$0.92) before provisioning, then budget all remaining preparation smoke, training, validation, inference, storage and contingency. Do not reuse full-KDA or238/462-fit estimates blindly: recompute this reduced28-flow/210-supervised/10-pretraining workload using actual prepared windows and measured throughputs. State measured versus projected terms, prices, GPU-hours and elapsed time. If the conservative estimate plus reserve exceeds remaining funds, stop before launch for a human decision.

Use Runpod skills, live stock/prices, compatible CUDA and explicit resource ownership. Stage checksummed immutable inputs and frozen scripts/manifests before long GPU allocation where practical. Start with measured assignments: small float64 flows/recurrent work on RTX2000 Ada-class hardware and large float64 flows on the measured Blackwell alternative when price/performance still supports it. Choose independent jobs per GPU; packing is optional only with measured aggregate benefit and adequate memory, not an assumed multiplier. No distributed training of one model required. Provision incrementally and monitor actual spend/progress against the cap; checkpoint and stop before an overrun. No idle paid resources during local development.

After correctness, learning and conservative budget gates pass, execute the reduced campaign under this vision; no additional routine approval round is needed. Material departures or blockers still require consultation. Keep user updates concise and milestone-based. Do not claim a complete comparison if required jobs fail; preserve failures and rerun only with explicit provenance and unchanged scientific contract.

Retain the richer output set rather than aggressively stripping hydrographs to save a few GB. Inventory all folds, variants, seeds, validation/primary/secondary periods and requested scores. Save best/final/pretrained weights and transforms, member and ensemble hydrographs, sampled/predicted parameters, metrics, learning curves, splits, manifests and provenance. Reuse immutable data where possible and distinguish compressed estimates from measured bytes. Prior23/51GB estimates are not definitive for this reduced workload. If approaching200GB, propose a retention/download plan rather than delete unique evidence or silently omit analyses.

Download locally, verify checksums and reader usability, then remove task-owned compute and temporary storage; confirm no remaining task spend. Preserve at least one verified durable copy throughout. Use the existing local evidence root or a clearly linked reduced-campaign subdirectory; do not overwrite prior smoke reports or present pilot scores as final results.

## Results and completion

Produce three agreed figures with no titles or explanatory prose inside the plots beyond axes/ticks and legends:

1. 2×2 flow scaling heatmap: median basin NPE of ensemble-mean discharge.
2. One CDF: validation-selected flow, alpha-HBV–LSTM, delta-HBV–LSTM, pure LSTM. Solid ensemble curves, thin transparent same-colour member curves, median basin NPE in parentheses in the legend. Selection may differ by regional fold; disclose this when interpreting the pooled curve.
3. Requested-versus-achieved for four flows, requests.50,.55,.60,.65,.70,.75,.80; median achieved ensemble NPE and diagonal reference. Requested calibration performance need not transfer exactly to later years or equal the score of averaged hydrographs.

Local evidence/data roots: `/Users/nicolaslazaro/Desktop/uzh-results/2026-11-11-regional-model-benchmark-setup/` contains scripts, `FOLLOWUP_REPORT.md`, frozen folds and prior measurements; sibling `2026-11-10-flow-training-examples/`, `2026-11-11-flow-subset-audit/`, `2026-11-11-regional-split/`, `2026-11-11-historical-climate-attributes/` contain verified raw examples, audit, mapping and climate provenance. Repository data are `/Users/nicolaslazaro/Desktop/work/hydrologeez/data/camels-us/`. Check existence, hashes and current core APIs. Do not recreate expensive upstream datasets unnecessarily.

Deliver a concise scientific summary with per-method/region results, selection rules, observation coverage, failures, actual settings/spend and limitations. The study is a reduced exploratory comparison, not an exact Feng reproduction or proof of general state of the art. Separate preprocessing advantages and transductive alpha access from architecture effects. Link all reproduction artifacts and update discussion#196. Use project uv, typed Python, Ruff, ty, focused test-first development and independent review for core changes. A blocked campaign must be reported honestly with recoverable artifacts and next decisions, not marked complete.
