# Soft NPE upper benchmark experiment

Experimental support for [#185](https://github.com/CooperBigFoot/hydrologeez/issues/185), returning findings to [#182](https://github.com/CooperBigFoot/hydrologeez/issues/182). No public metric or calibration API changes. Production adoption requires the published vision's human gate.

## Execution decision

On 2026-11-11 the user requested the full campaign on the same 16-core VM setup as the preceding normalising-flow experiment: Hetzner CCX43, 16 dedicated vCPUs, 64 GB RAM, hel1, **16 workers with one internal thread each**. This replaces the vision's six-worker local execution setting, not its scientific design. The published vision is not rewritten. Population 50, 69 generations, seeds 42–51, all six objectives and 50 fixed basins remain unchanged.

The local six-worker feasibility pilot used explicit basins 01013500, 09480000 and 14301000 twice each, with 50 uniform-bound parameter vectors per worker. All 3652 scored days were retained. Exact all-pairs sigmoid ranks use pair symmetry and O(n) auxiliary memory, without truncation, windowing, subsampling or changed rank approximation. Observation ranks and normalized FDC are cached. Full-period losses agreed with the existing Torch loss to 4.45e-16 absolute error. Measured soft-objective cost was approximately 22 ms per candidate; projected local six-worker campaign cost was 9.37 hours, excluding GA/startup/retries. This is not a VM timing claim.

## Reproduce

Use the repository's uv environment; install the `hybrid` extra for tensor correctness checks and the `cluster` extra (or `--with matplotlib --with seaborn`) for plotting. Freeze on the actual execution machine, after review and before calibration. Source, prepared data and installed package versions are hashed. Do not edit source or synchronize a different environment between freeze and run.

```sh
uv run --extra hybrid python -m scripts.soft_npe_pilot --output /path/pilot.json
uv run --extra hybrid python -m scripts.soft_npe_campaign trial --output /path/pilot
uv run --extra hybrid python -m scripts.soft_npe_campaign freeze --workers 16 --output /path/experiment --store data/camels-us/prepared --original-manifest /path/archived/manifest.json
uv run --extra hybrid python -m scripts.soft_npe_campaign run --output /path/experiment
uv run --extra hybrid --with matplotlib --with seaborn python -m scripts.soft_npe_campaign summarize --output /path/experiment
```

`trial` performs full-budget pilot trials for exact NPE and softness 0.1 on 01013500. It does not create campaign results. `freeze` samples the sorted archived 671-basin pool with `numpy.default_rng(42).choice(..., 50, replace=False)` and preserves its returned order. `run` uses the unchanged existing GA and HBV bounds/empty stores, forcing 1987-10-01 through 1999-10-01 exclusive and scores from 1989-10-01. Missing observations define a fixed mask; no basin is replaced.

Each trial atomically saves its own objective loss, exact NPE, parameters, candidate count, invalid candidates and runtime. Existing records are not overwritten; failures remain incomplete and are not silently retried. Preserve a failed record elsewhere explicitly before retrying its task. Use only one controller per output directory. If interrupted, execution receipts may remain `running`; completed trial records remain reusable and unfinished trials restart from their seed. No partial trial counts as complete.

`summarize` requires all 3000 successful trials and all 50 basins in every setting. Each setting selects its minimum own-objective loss across ten trials, never its best reported exact score. It writes selected parameters, paired basin differences, descriptive paired-trial differences, runtime/coverage, a report and a 300-dpi boxplot. No pooled best-soft objective is reported. Repeated trials are not independent basins. The selected score plot is in-sample and does not establish a higher theoretical maximum.

Local evidence and source snapshots: `/Users/nicolaslazaro/Desktop/uzh-results/2026-11-11-soft-npe-upper-benchmark/`. Red/green test logs and the numerical pilot are preserved there. The full campaign has not yet been launched. Review and a VM full-budget trial precede launch; publish findings to both issues and stop for the human decision.
