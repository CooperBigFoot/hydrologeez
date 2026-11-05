# Calibrate HBV ensembles

Calibration reuses [prepared CAMELS selections](camels.md) and the current HBV
runtime. It keeps bounded candidate discharge in memory, not candidate files.
One candidate has 13 parameters shared across a basin's zones; different basins
are calibrated separately. No rainfall, observation units, or QC flags change.

```python
from pathlib import Path
from hydrologeez.camels import Camels
from hydrologeez.calibration import CandidateEvaluator, Period, calibrate

inputs = Camels(Path("data/camels-us/prepared")).select(["01013500"], "1980-01-01", "2015-01-01")
evaluator = CandidateEvaluator(
    inputs.forcing,
    inputs.dates,
    inputs.discharge_mm_day[:, 0],
    0,
    Period("1981-10-01", "2000-10-01"),
    validation=Period("2000-10-01", "2015-01-01"),
    minimum_days=None,
)
ensemble = calibrate(
    evaluator,
    seeds=list(range(100)),  # 100 independent runs, not 100 candidates
    objectives=("kge_np", "nse"),
    population_size=100,
    generations=100,
    retain=5,
    batch_size=64,
    report=("training", "validation"),
)
# Parameters are in PARAMETER_NAMES order; regenerate only selected hydrographs.
solution = ensemble.solutions[0]
q = evaluator.discharge(solution.parameters[None, :])[:, 0]
```

For multiple basins, select their inputs once and construct an evaluator per
basin with the matching observation column and `basin_index`. Keep basin IDs
alongside the resulting ensembles. Seeds identify independent runs within each
basin. This example requests up to 500 models per basin, not a guaranteed count.
Use one objective and `retain=1` for 100 best-of-run models. The small local
[verification experiment](../planning/experimentation/2026-11-05-seeded-ensemble-calibration-for-hbv.md)
is a functionality demonstration, not a prescription for convergence.

## Search and retention

The locally owned real-valued GA/NSGA-II uses binary tournament selection,
elitist survival, bounded simulated binary crossover (distribution index 15),
and polynomial mutation (index 20, probability 1/13 per parameter). The GA
minimizes one loss; NSGA-II ranks non-dominated fronts and crowding for multiple
losses. Four seed-derived generators separate initialization, selection,
crossover, and mutation. Reproduction assumes the same code and NumPy version.

Every initial parameter is sampled independently, uniformly in its original
scale within the [maintainer's search bounds](hbv-light.md#maintainer-supplied-search-ranges).
Recession coefficients are **not** log-uniform. `bounds` can explicitly replace
these intervals in `PARAMETER_NAMES` order; physical HBV domains still apply.
Population size must be even and at least two. `generations=0` evaluates only
the initial population. Each run evaluates `population_size*(generations+1)`
optimization candidates; retained diagnostics are reevaluated separately.

Single-objective runs retain one best candidate. Multi-objective runs retain
up to `retain` unique parameter vectors from their final approximate front,
ordered by decreasing crowding distance with stable population-order ties.
Constant objectives contribute no crowding. Duplicate vectors are removed
within a run, without padding; equal objective scores do not imply equal
parameters. No distinctness is promised across runs, and no global front filter
replaces the requested per-run ensemble. These are alternative calibrated models,
not automatically samples from a probability distribution.

All [metric scores and components](metrics.md) are reported for requested periods,
independently of objective choice. Objectives may overlap or repeat. Signed biases
are optimized toward zero, not toward negative infinity. Undefined objective
values make a candidate ineligible for retention; an all-invalid run records zero
retained models. Check `ensemble.runs`, not just the requested count.

## Dates, missing data, and state

Every date interval is start-inclusive and stop-exclusive. Input dates must be
contiguous daily Gregorian ordinals. Each candidate starts with empty snow, soil,
groundwater and routing stores on the **first input day**, then simulates the
whole supplied record with its own parameters. Choose the input start earlier
than training to configure warm-up; no fixed warm-up length is imposed. Training
and validation cannot overlap; validation follows training. State continues
unchanged across the split and any unscored gap. Validation never steers selection.

Only observed missing days are excluded from scoring, never from simulation.
Candidate failures cannot improve scores by reducing their scored-day count.
`minimum_days` optionally rejects calendar years with fewer nonmissing observed
days **inside each scoring split**. Partial years and split-boundary years use
those actual counts, with no proration or automatic exemption. Accepted years
still omit missing days. `ensemble.coverage` shows pre-filter observed-day counts
for every year, including rejected years. No filter applies by default; a split
with no remaining observations raises an error.

## Memory and limits

`batch_size` bounds full-period discharge allocation (days × batch × 8 bytes),
plus simulation state, rank/sort workspace, and resident inputs. Exact Pool scores
need full scoring-period ranks and sorted flows; window-score averages would
change the objective. The optimizer retains only parameters and scalar scores,
with an O(population²) dominance matrix. There is no process-per-candidate pool;
Numba's own thread controls apply. The default batch 64 is a configurable
engineering choice, not a promised optimum or runtime limit.

The public result is an in-memory dataclass, not a new storage format. Retain
input basin identity, dates, bounds, settings and code/environment versions with
any exported results. No automatic campaign scheduling, checkpoints, or inference
of statistically independent/probabilistic ensemble members is provided.
