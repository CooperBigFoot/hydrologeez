# Discharge metrics

`hydrologeez.metrics.evaluate(observed, simulated)` evaluates aligned 1-D
arrays independently of calibration. It returns these scores plus component
diagnostics. All efficiency optima are 1; bias optima are 0.

| Name | Definition |
| --- | --- |
| `kge_np` | Pool (2018) non-parametric KGE: `1 - sqrt((beta-1)² + (alpha_np-1)² + (spearman-1)²)` |
| `kge` | Original KGE, replacing `alpha_np` with standard-deviation ratio and Spearman with Pearson correlation |
| `nse` | `1 - sum((sim-obs)²) / sum((obs-mean(obs))²)` |
| `log_nse` | NSE of natural-log flows with `mean(obs)/100` added to both series |
| `high_flow_bias` | FHV: `100 * (sum(sim_top)/sum(obs_top) - 1)`, independently sorted top 2% |
| `low_flow_bias` | FLV: `-100 * (A(sim)-A(obs))/A(obs)`, where `A(Q)=sum(log(Q_bottom)-log(min(Q_bottom)))`, bottom 30% |

Pool's `beta` is the mean ratio. `alpha_np` is
`1 - 0.5 * sum(abs(sort(sim)/sum(sim) - sort(obs)/sum(obs)))`.
Returned components are `beta`, `alpha_np`, `spearman`, `alpha` (standard-deviation
ratio), and `pearson`. Spearman uses average ranks for ties. FDC sorting removes
chronology; rank correlation does not measure explicit peak lag.

Tail counts are rounded **up** (`ceil(fraction*n)`), an explicit discrete
implementation convention. FLV measures anchored log shape, not low-flow volume:
a uniform positive rescaling has zero FLV. Zero low tails or a constant observed
low tail make FLV undefined; there is no artificial floor. Thus FLV cannot steer
calibration for many intermittent basins. Other metrics remain usable.

The log-NSE offset uses only the retained observations of the scored period.
It is the same for simulations and observations and scales with flow units.
It follows the documented Pushpalatha convention in hydroGOF, not log-KGE.

Only NaN observations are excluded. Negative or infinite observations raise an
error. Negative or nonfinite simulations on scored days invalidate all scores,
not the observation mask. Missing-day simulations do not affect scores. Fewer
than two observations, constant correlation inputs, and undefined denominators
produce NaN scores; identical constant series do not have perfect KGE.
`objective_loss(name, value)` minimizes `1-score` for efficiencies and absolute
percentage bias for FHV/FLV. Nonfinite scores yield infinity, never a reward.
Choose objectives for the scientific question; overlapping metrics are allowed.

## Sources

- [Pool, Vis & Seibert (2018)](https://doi.org/10.1080/02626667.2018.1552002),
  equations 1–7, pp. 1943–1944; p. 1945 motivates absolute FDC bias.
- [Pool et al. (2017)](https://doi.org/10.5194/hess-21-5443-2017) motivates
  targeted diagnostics, not a full ecological catalogue.
- [Yilmaz et al. (2008)](https://doi.org/10.1029/2007WR006716), FHV/FLV formulas
  checked against [NeuralHydrology](https://neuralhydrology.readthedocs.io/en/latest/api/neuralhydrology.evaluation.metrics.html).
  Unlike its implementation, no fixed flow floor is used here.
- [hydroGOF NSE documentation](https://search.r-project.org/CRAN/refmans/hydroGOF/html/NSE.html)
  documents the Pushpalatha (2012) observed-mean/100 offset. The primary
  Pushpalatha paper was not independently checked for this implementation.
