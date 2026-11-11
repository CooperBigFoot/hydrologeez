# Regional benchmark learning and GPU feasibility follow-up

Resolve the remaining feasibility questions before launching the regional model benchmark. Implement and verify KDA activation checkpointing, run bounded learning and operational pilots, and replace optimistic campaign projections with one measured scheduling, cost and storage recommendation. Stop for human approval before the full campaign. This is a focused follow-up, not a redesign or another model-selection tournament.

Inherit the scientific contract from [Regional model benchmark setup and GPU smoke tests](2026-11-11-regional-model-benchmark-setup-and-gpu-smoke-tests.md): 21 variants, seven HUC folds, 45 statics, 16 dynamic inputs, physical Daymet forcing and GA bounds, historical-only alpha pretraining, discharge MSE and exact NPE evaluation, ten comparator seeds/ten flow samples, 50 epochs (alpha 50+50), and validation-only selection. The 0.50–0.80 requested-score grid is a plotting range, not a training-data filter. No full 462-fit campaign, test-region tuning, time-varying parameters, static compression or changes to ensemble size are authorized.

The original setup delivered the standalone LSTM and explicit hybrid bounds in [PR #189](https://github.com/CooperBigFoot/hydrologeez/pull/189). Verify current main rather than reimplementing them. This standalone vision has no associated open discussion issue to close. Publication is not implementation or authorization to merge a core implementation PR.

## Evidence to reuse and claims to correct

Experiment scripts and evidence are local at `/Users/nicolaslazaro/Desktop/uzh-results/2026-11-11-regional-model-benchmark-setup/`, not bundled in GitHub. Read `REPORT.md`, `README.md`, `spec.py`, `training.py`, `smoke_suite.py`, `gpu_smoke/smoke_report.json`, `gpu_smoke/packing_probe.json`, `gpu_smoke/flow_50m_scale.json` and the telemetry. Prepared data paths `data/camels-us/...` are relative to the hydrologeez repository, not uzh-results. Preserve old evidence and clearly distinguish corrected conclusions from original reports.

Prior measurements on one RTX PRO 4000 Blackwell 24 GB establish useful starting points, not portable price/performance:

- Delta-LSTM step time changed from 3.37 s at B=32 to 3.65 s at B=1,024; throughput rose from 9.5 to 280.6 windows/s. Large physical batches help this implementation. Four B=256 workers achieved 286.7 aggregate windows/s, roughly the same as one B=1,024 worker, not another fourfold improvement over it. Dispatch overhead is a supported explanation, not proof of a “100%” bottleneck without profiling.
- Uncheckpointed trainable width-256 KDA consumed 17.99 GiB at B=32 and failed at B=64 for 1,095 days. Its proposed 16x memory reduction and campaign-cost savings are unmeasured. Frozen KDA is a different memory regime and must not substitute for trainable-path evidence.
- Float64 MAF avoids float32 rounding of 52,089 interior GA vectors onto bounds; preserve float64 data/model and recheck the contract. Small VRAM use does not establish cheap-GPU float64 throughput. The 1M flow packing probe measured approximately 73.6k, 72.3k and 71.4k aggregate examples/s for one, two and four workers: it did not show a benefit from packing. Resident synthetic 50M-row timings are not full end-to-end campaign timings.
- Per-update network-volume checkpointing can dominate short flow steps. Local container storage is not durable against pod loss.
- Existing cost estimates disagree and combine measurements with unverified hardware/optimization assumptions. The ~7 GB output estimate also needs a complete output inventory.

The prior run reports $0.31 spent and deletion of its pod and temporary volume. Recheck current account/resource state before acting; do not recreate deleted IDs or touch unrelated resources.

## Implement and prove KDA activation checkpointing

Add the smallest coherent optional segment activation-checkpointing support to reusable KDA in hydrologeez. Sixty-four-day segments are a starting candidate, not a scientifically required constant. Preserve architecture, parameter count, physical model and full-sequence gradient propagation; do not detach segment boundary states or truncate backpropagation. Frozen/inference execution should not pay needless recomputation overhead. Keep checkpoint configuration and saved-model reconstruction explicit where relevant.

Use test-first development and independent reference comparisons: forward outputs, final states where exposed, input gradients and parameter gradients versus uncheckpointed execution over multiple segments and a remainder, including actual training-stage use. Require justified numerical tolerances rather than bitwise equality. Check freezing, save/reload and real CUDA backward behavior. Record precision, sequence length, batch, segment length, peak memory and synchronized timing. Checkpointing trades recomputation for memory: do not promise speedup until measured.

Benchmark representative trainable delta-KDA and alpha-pretraining KDA, plus frozen alpha inference/training as needed. Measure realistic large physical batches and compare with the original path. No exhaustive segment-size sweep is needed. If this does not make the workload feasible, report the blocker and alternatives without changing width, physics, sequence objective or seeds silently.

## Verify the training data and learning opportunity

Audit the implemented finite sequence-window dataset before using it for projections. The proposed default is 1,095 days, 365 warm-up, 730 scored, stride 365. Check actual calendar dates, leap years, tail anchors, coverage, overlap multiplicity and missing-observation masks. Every supervised loss element must lie in 1989-10-01 through 1999-09-30; alpha pretraining ends before 1999-10-01. Explain how initial warm-up differs from subsequent overlapping windows and the historical context used for fixed hybrid parameters. No test discharge enters fitting or selection.

One epoch means one pass through this explicit dataset (or selected flow rows). Retain partial final batches; fewer examples than the nominal batch size does not imply zero updates. Verify valid-element-weighted MSE under unequal microbatches and missing targets, gradient-accumulation normalization, last partial accumulation and scheduler/update counts. State optimizer steps per epoch and per 50-epoch run. Roughly 250 large-batch updates may or may not suffice; finite gradients alone are not evidence of useful training.

Run a short real-data learning pilot for each distinct family/stage: flow, standalone LSTM, delta-LSTM, delta-KDA, alpha-LSTM pretraining/frozen training and alpha-KDA pretraining/frozen training. Use a bounded representative development fold/subset and seeds, not all configurations/folds. Cover materially different flow sizes/data scales where necessary to judge training opportunity. Record training and validation curves and meaningful predictor changes, preserving evidence of flat, unstable or diverging behavior. Alpha reconstruction improvement alone does not establish useful frozen representations; inspect downstream validation too. Do not require monotonic improvement or a particular NPE to claim a technically valid pilot.

Use the existing proposed defaults as pilot starting points: stratified 15% whole-basin validation manifest; four MAF transforms with three-layer widths 60/92/172/332; nested float64 sampling; softplus standalone readout; Adam 1e-3 with cosine decay to 5e-5. Verify their actual implementation and input isolation. Do not add a hyperparameter tournament. Document flaws or inadequacy and recommend changes for human review, rather than silently replacing the agreed 50-epoch campaign or claiming short pilots prove final convergence.

## Durable recovery, inference and artifact accounting

Revise the per-update persistence policy for fast flows based on measured overhead. Periodic atomic durable checkpoints are authorized for this follow-up; 500 updates is a starting candidate, not a requirement. Local NVMe may stage writes but is not the only recovery copy. Retain best-validation and final checkpoints. Clearly state the maximum work lost under the recommended interval and distinguish process restart from pod/storage loss.

Prove checkpoint recovery in a separate process from the durable copy: optimizer/scheduler, RNG, sampler/data position, next update and saved model behavior. Include partial batches and accumulation boundaries; either save only completed updates or correctly preserve partial accumulation. Test interrupted writes without destroying the only valid checkpoint. Measure serialization, device synchronization and durable-write overhead within realistic training timing, not only isolated compute steps.

Verify inference can run later from saved artifacts. Representative smoke inference is permitted, but test results must not steer model choices; use validation/development data for learning judgments. Preserve all output contracts from the first vision.

Recompute storage from an explicit inventory: every fold/configuration/seed, retained and temporary checkpoints, alpha pretraining states, ten member hydrographs and ensemble, both evaluation periods, all requested scores needed for steering and validation selection, static parameter vectors, metrics and provenance. State what is recomputed instead of persisted and what is shared rather than duplicated. Give uncompressed sizes and measured compression estimates separately; do not assume final output is 7 GB or downloads take less than ten minutes. Propose selective retention/download only if needed, without losing unique evidence or silently changing requested analyses.

## Hardware comparison and bounded spending

Independent jobs per GPU remain the strategy; no distributed training of one model by default. Prefer the simplest measured efficient assignment: one large physical-batch hybrid job or several smaller independent jobs if aggregate end-to-end throughput justifies packing. Accumulation increases effective batch size, not physical GPU utilization by itself.

Benchmark at least one cheaper GPU candidate against the existing measured reference under equivalent inputs, float64 flow precision, optimizer work and timing boundaries. Check live Runpod price, stock, CUDA compatibility and storage placement using the repository skills. Do not infer throughput from hourly price, memory size or single-precision specifications. Include representative flow and recurrent workloads; report candidate failures and lack of capacity honestly. Remeasure reference paths if changed code makes old timings incomparable. No broad GPU tournament.

Use bounded paid smoke/learning pilots only. The $170 ceiling covers the whole project, including prior smoke spend, this follow-up, future training, inference and storage; account balance is not permission to raise the ceiling. Before provisioning, state the proposed runtime/spend cap and remaining reserve. Obtain human approval if the necessary pilot materially consumes the campaign budget. Do not leave paid GPUs idle during development or downloads. Verify cleanup and checksum-verified retrieval, preserving unique artifacts before deletion; only manage resources created for this task.

## Deliver one decision-ready recommendation

Keep reusable changes in hydrologeez and experiment assembly/evidence in uzh-results. Use uv, strong typing, Ruff, ty and focused tests. Report core defects and experiment-runner bugs with minimal reproduction, expected/observed behavior, impact and proposed repair. This vision authorizes the KDA checkpointing implementation and necessary focused runner corrections; material scientific changes or unrelated core repairs require explicit review. Obtain independent review of core changes through normal PR process; publication of this vision does not itself merge those changes.

Deliver a concise corrected report linking detailed evidence, with:

- Correctness/recovery outcomes and open blockers, not just successful commands.
- Learning curves and whether the proposed schedule has a credible learning opportunity, with limitations.
- A concrete assignment of model families/jobs to GPU types, physical/effective batch sizes, accumulation, packing and persistence cadence.
- One internally consistent campaign estimate including alpha pretraining, all fits, data loading, validation/selection, final inference, checkpointing, provisioning overhead and storage. Separate observed rates from extrapolated quantities and unmeasured terms. Show both elapsed time and aggregate GPU-hours, price timestamps, expected range and contingency; never claim the campaign fits the ceiling solely from an optimistic endpoint.
- Complete output-size and download/retention plan, actual follow-up spend, resource cleanup receipts, and the exact remaining human decisions.

Preserve prior reports as evidence, but mark which conclusions the corrected report supersedes. A failed or inconclusive pilot is a valid result when it yields a reproducible blocker and a clear next decision. Stop without launching the full campaign, even if measurements appear favorable.
