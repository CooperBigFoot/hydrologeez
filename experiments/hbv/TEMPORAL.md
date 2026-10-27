# Discrete Newton soil feasibility probe

## Result

Do not extend this implementation to the full model yet. Of 24 isolated soil
cases, 8 converge and satisfy the original recurrence and reference comparison;
12 stop at an invalid first Newton trial and 4 stop at a dry fractional-power
singularity. All successful cases are slower than compiled sequential soil.
This is evidence about a bounded NumPy CPU probe, not a rejection of temporal
methods or a prediction of GPU performance. Nothing is adopted into production.

## Method and scope

`uv run --group experiments python -m experiments.hbv.temporal`
regenerates `evidence/temporal.json`. Run tests with
`uv run --group experiments pytest tests/test_hbv_temporal.py`.

The exact ordered soil map follows `docs/hbv-light.md`: recharge from old moisture,
capacity overflow, then capped moisture-limited evaporation. Each Newton step
linearizes this map at a guessed trajectory and solves the resulting scalar affine
recurrence by a real Hillis–Steele prefix scan. Composition is
`(a2,b2) ∘ (a1,b1) = (a2*a1, a2*b1+b2)`. Work is O(T log T) per iteration,
with O(T × batch) memory, not a dense trajectory Jacobian. Threshold derivatives
select an active branch; no smooth-convergence theorem is claimed there. At exact
capacity equality the selected slope is zero; at LP equality the demand slope
is zero; at equality of available water and evaporation demand the final-state
slope is zero. These are one-sided/active-set choices, not unique classical
derivatives. Rejecting singular dry recharge even when a later cap might cancel
it is a conservative limitation of this probe.

Method reference: Lim, Zhu, Selfridge & Kasim (2024), *Parallelizing non-linear
sequential models over the sequence length*, ICLR 2024,
[arXiv:2309.12252](https://arxiv.org/abs/2309.12252), discrete Newton formulation
and affine scan (section 3.4, Appendix A.1). The paper does not establish global
convergence for these nonsmooth HBV rules; these unsafeguarded failures do not
disprove DEER.

Inputs are explicitly synthetic **isolated supplied soil I**, not a generated snow
trajectory or a full HBV run. I and PET are daily mm, aligned by day and independent
column. No warm-up is excluded; all days use the given candidate parameters and
initial memory. Arrays retain all end-of-day M, recharge R and evaporation A.
Both methods materialize the same requested outputs. No routing, whole-model
restart or basin discharge performance is claimed. A soil continuation test checks
restart from its sole persistent state.

## Settings and acceptance

- Float64 CPU NumPy scan versus Numba-compiled sequential soil; local Apple arm64.
- 365 and 3650 days, batches 1 and 8; FC=100 mm, LP=0.5. These are experiment
  constants, not scientific defaults. Seed 90210, five measured repetitions.
- The repeated initial state is the uninformed trajectory guess. No reference
  solution is used for initialization. No clipping, damping, smoothing or fallback.
- Hard cap: 32 Newton steps. Original recurrence residual must be at most
  1e-10 mm; all M/R/A errors must be at most 1e-8 mm against compiled sequential.
  These conservative float64 absolute tolerances allow accumulation beyond unit
  roundoff while remaining far below the millimetre scale; they are not a bound
  valid for arbitrary durations/parameters. Relative errors divide by at least
  1e-12 mm to make zeros explicit rather than undefined.
- Reject every nonfinite or out-of-[0,FC] trial, even tiny excursions. Singular
  derivative at M=0, I>0, BETA<1 stops honestly. For I=0 the recharge derivative
  is exactly zero, avoiding artificial `0*infinity`. A residual-zero trajectory
  needs no derivative, even at a dry state.
- Stress cases use Gamma(1.2, 5) I, 30 dry days per 90, 150 mm pulses, sinusoidal
  PET and periodic 100 mm PET; BETA=2,1,0.5 and M0=50, plus BETA=0.5/M0=0.
- Threshold case repeats I=(50,0,150,0,0), PET=(0,2,0,100,0), BETA=1/M0=0.
  It covers LP equality, capacity, overflow, dry states and water exhaustion.
- Separately labelled smooth control uses I=3, PET=2, BETA=2/M0=50. Added after
  stress failures to distinguish implementation viability from robust convergence;
  the failed cases remain in the evidence and no solver setting was changed.

## Measurements

Median wall times include guess initialization, all map/derivative evaluations,
scans, residual checks and output allocation. NumPy is eager CPU execution: no
GPU transfer or deferred synchronization. Input preparation and first compiled
baseline call are separate JSON fields. No disk recording is timed. Component
substage times are not individually instrumented; the aggregate includes them.
RSS is process peak over this small matrix, including interpreter/compiler and
references, not solver-exclusive allocation. Hardware and all five samples are
in the evidence. There was no concurrent CPU/GPU timed worker.

| Converged case | Days | Batch | Iterations | Sequential ms | Newton ms | Slowdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| steady_nonlinear | 365 | 1 | 4 | 0.012 | 0.166 | 14.17× |
| thresholds | 365 | 1 | 2 | 0.007 | 0.084 | 12.89× |
| steady_nonlinear | 365 | 8 | 4 | 0.085 | 0.353 | 4.15× |
| thresholds | 365 | 8 | 2 | 0.047 | 0.183 | 3.88× |
| steady_nonlinear | 3650 | 1 | 4 | 0.103 | 0.452 | 4.38× |
| thresholds | 3650 | 1 | 2 | 0.059 | 0.270 | 4.55× |
| steady_nonlinear | 3650 | 8 | 4 | 0.829 | 2.842 | 3.43× |
| thresholds | 3650 | 8 | 2 | 0.472 | 1.398 | 2.96× |

Smooth controls converge in four iterations: largest state error 1.92e-13 mm,
recharge error 6.44e-15 mm, evaporation error zero; largest original recurrence
residual 4.98e-14 mm. Threshold cases converge in two iterations with zero error
and residual. These successes satisfy the declared fidelity thresholds.

Stress cases with M0=50 all stop after their first invalid trial. Dry fractional
cases stop before a Newton step at the singular derivative. JSON retains errors
and residuals of their last valid guesses, plus invalid trial-state counts. Those
values are **not solutions**; their fast early-exit times are never speedups.
Failure rate is 16/24 workloads (66.7%); whole batches fail if any column fails.
This is not a sampled population failure estimate. None reaches the cap; a test
checks that capped unconverged guesses are labelled failures.

## Limits and next decision

NumPy allocation and O(T log T) scan work hurt these short, cheap scalar recurrences.
A work-efficient GPU scan could behave differently. Domain failures remain a
separate mathematical problem, not merely launch overhead. Safeguarded or
semismooth Newton, informed warm starts and GPU scans would be new investigations
with their own costs and fidelity checks. This probe deliberately stops without
building that machinery. The independent compiled sequential baseline remains
more useful for the tested inputs.
