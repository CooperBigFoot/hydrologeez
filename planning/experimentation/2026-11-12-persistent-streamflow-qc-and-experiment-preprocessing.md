# Persistent streamflow QC and experiment preprocessing: experimentation notes

Vision: [Persistent streamflow QC and experiment preprocessing](../visions/2026-11-12-persistent-streamflow-qc-and-experiment-preprocessing.md)

## Streamflow QC

### Homogeneity critical values beyond n = 100

Question: GSIM Part 2 applies the four ECA&D homogeneity tests at the 1% level with the
ECA&D ATBD (2013) critical-value tables, which stop at n = 100. Most monthly index series
have more reliable values. What should n > 100 use?

Approaches:

- Hold the n = 100 values constant. The Pettitt critical value grows roughly as n^1.5, so
  this would reject nearly every long monthly series.
- Use asymptotic formulas. These exist for Pettitt, Buishand and von Neumann, but not in a
  simple form for SNHT, and they jump away from the published values at n = 100.
- Simulate iid normal series with the implemented statistics, as the ATBD did for its tables.

Observations (`uv run python -m scripts.homogeneity_critical_values --check`, 100 000
replicates per n):

- Simulated SNHT T0 (population SD), Buishand R/sqrt(n) and von Neumann N reproduce every
  published 1% row within simulation error. For example, n = 20 gives 9.56, 1.604 and 1.039
  against 9.56, 1.60 and 1.04.
- The published Pettitt "1%" values match the simulated 98th percentile of max|Xk|, not the
  99th: n = 20, 50 and 100 give 70, 292 and 840, against 71, 293 and 841. The 99th
  percentile gives 75, 313 and 903. The table is therefore a one-sided 1% level.

Decision: keep the published rows for 20 ≤ n ≤ 100. Add simulated rows for n = 150...5000,
using the same quantiles as the published table (the 98th percentile for Pettitt). Interpolate
linearly in n, which is mildly conservative on the convex Pettitt curve and clamps above
n = 5000. The rows and the script that produced them are in the repository.

### Sparse seasonal or monthly subseries under detrending

Question: GSIM Part 2 removes a least-squares linear trend (`x = b + a t`) separately from each
season or month subseries whenever the whole index series has at least 20 reliable values. How
should a season or month with only 1 or 2 reliable values be treated?

Observations: a 2-parameter line fits 1 or 2 points with zero residuals (`[0.]` or `[0., 0.]`),
injecting artificial zeros into the AR(1) likelihood and the four homogeneity statistics even
though the subseries had no residual degrees of freedom.

Decision: require at least 3 reliable values in a season or month subseries before fitting its
trend; set sparser subseries to missing after detrending and re-check the 20-value minimum and
residual constancy before pre-whitening.

### Power of the prescribed pre-processing

Question: do detrending and AR(1) pre-whitening still detect the shifts that the tests target?

Observations (synthetic 40-year monthly series with a seasonal cycle, a trend, AR(1) noise
and 40 unreliable months, with a shift after one third of the record):

- AR(1) noise with phi = 0.3: a shift of 3 innovation SDs was "suspect" for all seeds tried,
  and the series without the shift was "useful". Raw tests on the undetrended series rejected
  three or more tests even without the shift.
- AR(1) noise with phi = 0.7: shifts of up to 5 innovation SDs stayed "useful". The per-month
  trends absorb part of the step, and pre-whitening reduces the rest by about (1 − phi).
- Yearly white noise, n = 40: even a 6 SD shift after one third of the record stayed "useful",
  because the fitted linear trend absorbs most of the step.

Decision: implement the procedure as published. The paper notes both effects: pre-whitening
reduces the jump, and detrending creates a spurious trend. Read "useful" as "not detected
under the published procedure", not as evidence that the series is homogeneous. Tests use
the phi = 0.3 monthly case.
