# Seeded ensemble calibration for HBV

Build a practical calibration engine for the existing HBV runtime and prepared
CAMELS inputs. The user wants ensembles of 100 or more calibrated models per
basin, with selectable objectives and fast repeated evaluation. Preserve the
existing hydrological behavior; runtime replacement is not the task.

Discussion: [#118](https://github.com/CooperBigFoot/hydrologeez/issues/118).
Delivery and verification of this calibration capability completes #118's design
and implementation handoff; publication alone does not complete or close it.
The broader throughput research in #89 and Program #87 remains separate.

## Independent runs, populations, and retained models

Keep these concepts separate:

- Runs per basin: independent seeded genetic-algorithm calibrations.
- Population size: candidates evolved inside a run.
- Retained solutions: models contributed by that run to the ensemble.

Initialize every run's population by independently sampling each parameter
uniformly in its original scale within the maintainer-supplied search bounds in
[HBV-light](../../docs/hbv-light.md#maintainer-supplied-search-ranges). These are
search bounds, not model validation limits. Do not silently use log-uniform
sampling for recession parameters. Evolution changes the initial distribution;
keep evolved candidates within the selected bounds. Preserve seeds and run
identities so experiments can be reproduced and results traced to their runs.

Support single-objective and multi-objective calibration. The user chooses the
metrics and is responsible for selecting a scientifically sensible combination;
do not restrict combinations because metrics overlap or fail to compete.
Single-objective runs can contribute their best parameter set. Multi-objective
runs produce an approximation to a Pareto front and can contribute multiple
non-dominated solutions. Examples are 100 runs retaining one best model each,
or 100 runs retaining up to five front members each. Population members are not
implicitly calibrated ensemble members.

The precise policy for selecting a limited number of front members is not
settled. Choose and document a simple, explicit policy, preserving objective
values and run provenance. Do not silently duplicate solutions to reach a count
or promise distinct solutions when objectives collapse. A solution on one run's
front may be dominated across runs; do not silently replace the requested
per-run ensemble with a globally filtered front. Such ensembles are alternative
calibrated models, not automatically samples from a probability distribution.

## Objectives and diagnostics

Implement Pool et al. (2018)'s non-parametric KGE faithfully. It combines the
simulated/observed mean ratio, normalized flow-duration-curve agreement, and
Spearman rank correlation using distance from their ideal values. Retain useful
component diagnostics as well as the composite score.

Provide a small, clearly defined metric surface usable both during calibration
and for independent evaluation of simulations. Original KGE, NSE, log-NSE, and
high-/low-flow bias measures are the intended starting candidates alongside the
required Pool metric. Exact tail definitions, log zero treatment, and the final
small supporting catalogue remain implementation design work: ground them in
sources, document conventions, and validate them rather than silently guessing.
Do not expand into an ecological streamflow-characteristic catalogue. Explicit
peak/event-timing metrics are deferred. Log-KGE is not a required deliverable;
Pool (2018) warns about its near-zero and unit sensitivity.

A flow-duration curve ignores chronology. Rank correlation measures daily
association, not explicit peak lag. Signed bias is useful diagnostically, but
optimization must target agreement, not reward arbitrarily negative bias.
Make objective direction and optimum clear. Handle ties, zero flows, missing
observations, and undefined scores consistently without rewarding invalid
candidates. Report selected diagnostic metrics independently of which metrics
steer the optimizer; do not prescribe one scientifically preferred combination.

Sources:
- Pool, Vis & Seibert (2018), *Evaluating model performance: towards a
  non-parametric variant of the Kling Gupta efficiency*,
  https://doi.org/10.1080/02626667.2018.1552002 (equations 1–7).
- Pool et al. (2017), *Streamflow characteristics from modeled runoff time series
  – importance of calibration criteria selection*,
  https://doi.org/10.5194/hess-21-5443-2017. This motivates targeted evaluation,
  not implementing its full ecological signature catalogue.

Both PDFs are available locally under `~/Downloads/Pool_etal_2018_HSJ_*` and
`~/Downloads/Pool_etal_2017_hess-21-5443-2017.pdf`.

## Periods and observations

Allow configurable warm-up and training/validation date splits. Optimize on the
training period; report performance on training, validation, or both as requested.
Validation is held out from optimization. Exclude warm-up from scores and evolve
each candidate using its own parameters; do not reuse another candidate's warmed
state. Make initialization and split-boundary state handling explicit.

Exclude missing observations from evaluation without filling them. Simulate
continuously through gaps and excluded scoring years to preserve state. Use an
observation-based mask shared by candidates, not a candidate-dependent mask that
can improve scores by dropping failed simulations.

Offer optional year-level coverage filtering with a configurable minimum number
of nonmissing observed days. Accepted years still exclude their missing days;
rejected years contribute no scores. Document the year convention and treatment
of partial years and split boundaries. No numerical threshold, mandatory calendar
convention, or new QC filter was selected. Keep coverage visible in results.
Prepared data retain original QC and use the CAMELS forcing-header area for
observed mm/day; do not silently change those preparation decisions.

## Reuse the current runtime; evaluate in memory

The natural home for orchestration is `src/hydrologeez/calibration`, without
forcing metrics into an optimizer-only API. Reuse inputs across candidate
populations and repeated runs. Calibration should not require saving candidate
hydrographs to disk: retain scores, parameters, identities, and useful run
information; hydrographs can be regenerated for retained models.

Repository facts to verify when implementing:
- [CAMELS inputs](../../docs/camels.md) are locally prepared at
  `data/camels-us/prepared`, with 671 basin identities and aligned observations.
  Selection materializes reusable forcing and observations.
- `simulate(history=False)` already returns in-memory discharge without other
  retained histories. Simulation rows share forcing through `basin_index`.
- Complete final memory supports time-window continuation with unchanged
  parameters. The [output writer](../../docs/hbv-output.md) is not required for
  calibration; its 256-row/365-day defaults are not calibration requirements.
- NSE and ordinary KGE permit running-statistic accumulation. Exact Pool
  rank/FDC scores require information over the whole scoring period; averaging
  independently calculated window scores changes the objective.

A useful starting hypothesis is to keep full-period discharge for a bounded
candidate batch, score it, then discard it. For illustration, 256 candidates ×
12,784 days × float64 needs about 25 MiB for discharge alone, excluding sorting
workspace, inputs, and other memory. This is not a mandated batch size or a
throughput claim.

`../ctrl-freak` provides real-valued GA and NSGA-II with population-level
`evaluate_batch` callbacks, seeded operators, and bounded crossover/mutation.
Reuse that approach where helpful. Dependency adoption versus a small local
implementation is delegated; do not copy its full framework without need.
Avoid assuming process-per-candidate execution improves an already batched,
Numba-parallel simulator.

## Experiment, measure, and iterate

The implementation agent is explicitly authorized to think, implement small
experiments, inspect evidence, and iterate toward an effective implementation.
Do not treat the first design as final or attempt a one-shot implementation.
Batch sizes, time windows, scoring layout, observation precomputation, optimizer
integration, and scheduling are reversible engineering choices, not fixed here.
Use proportionate experiments to choose among plausible approaches rather than
an exhaustive optimizer/backend tournament. No cloud provisioning is authorized.

Separate correctness from performance. Compare equivalent objectives and outputs,
account for setup/JIT, input reuse, simulation, metric computation, and optimizer
overhead, and report representative end-to-end time and memory with workload and
measurement limits. Existing disk-output timings do not establish calibration
throughput. Seek measured improvements, not a claim of global optimality.

Follow project test-first guidance, use `uv`, and validate with Ruff, ty, and
appropriate tests. Meaningful evidence includes independently checked metric
values, reproducible seeded runs, valid parameter bounds, correct train/validation
and missing-data behavior, retained-solution provenance, and equivalence of scores
across evaluation batch choices. Demonstrate a small end-to-end prepared-data
calibration and document how to request 100+ runs; no full 671-basin × 100-run
production campaign is required merely to prove the interface.

Success is a documented, usable calibration path with bounded in-memory candidate
evaluation and evidence supporting the chosen implementation. Keep remaining
scientific conventions visible, and report limitations rather than hiding them
behind defaults. This vision authorizes implementation through the separate
implementation workflow, not automatic execution on publication.
