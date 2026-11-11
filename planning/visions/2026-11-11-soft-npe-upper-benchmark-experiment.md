# Soft NPE upper benchmark experiment

Determine whether calibrating HBV against soft NPE finds better **in-sample exact NPE** than calibrating directly against exact NPE under the same search budget. Implement the bounded experiment, run it, and report results with a clear boxplot. No held-out prediction evaluation is wanted.

This vision first delivers experiment [#185](https://github.com/CooperBigFoot/hydrologeez/issues/185), then pauses at a mandatory human gate before any production adoption. Return findings to the originating discussion [#182](https://github.com/CooperBigFoot/hydrologeez/issues/182). The agent assesses the numerics and presents the plot to the human; together they decide whether soft NPE is worth adding to hydrologeez. Publication alone completes neither issue. Experimental objective support is authorized immediately upon implementation of this vision; full integration is conditional on explicit human approval after the experiment. Changes to exact NPE and hybrid training experiments are outside scope.

## Comparison

- Randomly sample 50 basins without replacement from the existing 671-basin CAMELS-US pool. Save the selection seed and basin IDs. Use the same basins for every setting; disclose unusable basins or failures rather than silently replacing poor results.
- Reuse the existing local, individual-basin HBV calibration setup: 13 parameters and existing search bounds, forcing/PET preparation, and empty initial stores. Simulate from 1987-10-01 through 1999-10-01 exclusive. Score 1989-10-01 through 1999-10-01 exclusive, after two years of warm-up.
- Use the existing genetic algorithm for both objectives, population 50 and 69 generations: 3,500 candidate evaluations per trial. Run ten trials with seeds 42–51 per basin per setting. Pair seeds across settings and hold all non-objective settings fixed.
- Compare exact NPE against soft NPE at softness **0.01, 0.03, 0.1, 0.3, and 1.0**. These are an exploratory grid, not established optimal values. Do not introduce another optimizer or adapt the grid after inspecting results without identifying that as follow-up work.
- Use **six workers** for the experiment, with internal threading managed to avoid nested oversubscription.
- Each setting selects its best candidate across its trials using its own calibration objective. Evaluate those selected candidates with unchanged exact `kge_np`. Each plot box contains one exact-NPE score per basin, not ten seed scores treated as independent basins. Preserve per-trial results as well.

This is 10.5 million candidate-basin evaluations across six settings. Equal candidate budgets isolate the objective comparison, not equal compute costs. Record elapsed runtime and failures, including attempted and successful trials per basin and setting. Partial trials do not count as a completed equal-budget comparison; report them explicitly before drawing conclusions. Any improvement demonstrates better finite-budget search results, not an increase in the theoretical maximum of exact NPE.

## Existing evidence and implementation scope

The completed experiment archive is `/Users/nicolaslazaro/Desktop/uzh-results/2026-11-08-nearest-donor-joint-calibration/`, particularly `README.md`, `RESULTS.md`, and retrieved calibration configuration/results. It supplies the existing 671-basin, ten-trial, 3,500-evaluation setup and scoring dates. The sibling path from this checkout is **../../uzh-results**, not ../uzh-results. The later `2026-11-10-flow-training-examples/` archive proposes a different 10,100-evaluation budget; do not substitute that budget here. Inspect archived settings and data availability before running, and record code/data provenance rather than assuming archived scores can be reused unchanged.

Jan Seibert, Marc Vis and Sandra Pool's [Setting the bar: benchmarks for model performances in large-sample hydrology](https://doi.org/10.5194/hess-30-5857-2026) motivates this local upper benchmark. The paper is at `/Users/nicolaslazaro/Downloads/hess-30-5857-2026.pdf`; printed pages 5859–5860 describe in-sample calibration/evaluation and ten 3,500-run trials. This experiment follows that idea, not a claim of exact reproduction of every published forcing or model setting.

[#181](https://github.com/CooperBigFoot/hydrologeez/issues/181) already delivered `hydrologeez.losses.soft_npe_loss`. Preserve its mathematical meaning: unchanged mean-flow and normalized flow-duration components, exact observed midranks, and sigmoid soft predicted ranks at temperature `softness * mean(observed)`. Use exact NPE only for the common reported scores. Before the human gate, keep experimental calibration support narrow; do not add soft variants to the public metric catalogue or refactor unrelated calibration machinery.

The current all-pairs implementation is quadratic in scored days. First check correctness and full-period runtime/memory on a small representative pilot before launching the campaign. An equivalent efficient implementation is allowed, with numerical agreement checks against the existing loss. Do not silently substitute a different rank approximation, subsample days, or average short-window NPEs: those change the experiment. Cache invariant observation-side work where useful. Treat invalid candidates consistently and never reward undefined scores. If the planned run is impractical with six workers, report measured evidence and the blocker before changing the agreed budget or objective.

Leave reversible execution, storage, and implementation details to the implementing agent. Use project-native uv execution and the repository's test-first, typing, Ruff and ty guidance for code changes. Keep experimental code reproducible without turning it into a general benchmark framework.

## Results and figure

Save the experiment's scripts/configuration, basin selection, per-trial and selected scores/parameters, runtime/failure records, report, and figure in a clearly named experiment directory under `/Users/nicolaslazaro/Desktop/uzh-results`. Preserve existing archives. Record enough settings and code/environment versions to rerun the comparison; avoid storing every candidate hydrograph unnecessarily.

The approved visual reference is `/Users/nicolaslazaro/Desktop/npe-calibration-mock.png`, which uses synthetic data only. The real figure must have:

- Six boxplots in order: Exact NPE, then Soft NPE 0.01, 0.03, 0.1, 0.3, 1.0. Put softness on a second tick-label line if useful.
- Exact NPE on the y-axis, labelled `NPE`. The completed comparison requires the same 50 basins' valid exact-NPE scores in every box. If failures prevent that, preserve the partial results and report the blocker rather than fabricate values, replace basins, or present a smaller sample as the completed experiment.
- A grey baseline and a different colour for each soft variant.
- No title, legend, x-axis label, or other text beyond axis labels/tick annotations. Use `sns.despine()`.
- Axis limits that show all actual results; do not blindly copy the mock's limits. Save a high-resolution PNG and reproducible plotting code.

The report should summarize the distributions and paired per-basin differences from the exact-NPE baseline, including win counts and mean/median changes for each softness. Keep the softness settings separate rather than hiding their extra search budget in a pooled best-soft result. Describe whether gains are consistent across basins and trials, runtime costs, failures, and limitations. A negative result on this grid does not rule out every possible surrogate. Do not claim statistical independence for repeated trials within a basin.

## Human gate and conditional integration

Report back to the human with the plot and a concise numerical assessment: improvements by softness, paired gains and losses, consistency across trials, and compute cost. Link the saved artifacts and return findings to #185 and #182. Stop here and wait for explicit human approval; a positive numerical result or agent recommendation is not permission to proceed.

The human and agent decide together whether the benefit warrants full integration, and which softness setting or configuration should be carried forward. Record that decision. If rejected or deferred, preserve the experiment and stop without production changes. The completed research can close #185; it does not by itself complete #182.

If the human explicitly approves adoption, this same vision authorizes the agent to integrate soft NPE fully into hydrologeez as an additional metric and calibration objective alongside the existing ones. Preserve exact NPE and its current behavior, distinguish the soft metric by name, reuse shared numerical semantics rather than duplicate competing definitions, and expose the agreed softness configuration. Cover numerical correctness, public evaluation/calibration use, invalid-input behavior, and relevant persistence/configuration paths with meaningful tests. Document how to use it and its limits; follow normal implementation review and validation before delivery. Leave precise API mechanics to the implementing agent. Resolve any material adoption-scope changes with the human at the gate rather than assuming the experiment settles them. Full approved integration, not the experiment alone, can complete #182.
