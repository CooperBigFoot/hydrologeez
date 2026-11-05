# Seasonal climate-regime clustering

Install with `uv sync --extra cluster`. This Python workflow groups complete-year
seasonal climatologies, not coincident weather events or hydrological responses.
No discharge coverage filter is applied. The full CAMELS study remains [#142](https://github.com/CooperBigFoot/hydrologeez/issues/142);
attribute interpretation and classifiers remain [#141](https://github.com/CooperBigFoot/hydrologeez/issues/141).

## Prepared CAMELS example

```python
from pathlib import Path
from hydrologeez.camels import Camels, read_attributes
from hydrologeez.cluster import camels_cycles, run_experiment, load_experiment

store = Camels(Path("prepared/daily"))
ids = list(store.ids[:12])  # small exploratory experiment, not the full study
cycles = camels_cycles(store, ids, "1980-01-01", "2015-01-01")
attributes = read_attributes(Path("prepared/static"), ids=ids)
coordinates = attributes.table.select("gauge_lon", "gauge_lat").to_numpy()
result = run_experiment(cycles, Path("results"), "pt-small", k=3, seed=42, coordinates=coordinates)
basin_ids = result.clusters[0]  # string IDs, including leading zeros

# Explicit search instead of k; no implicit count range.
search = run_experiment(cycles, Path("results"), "pt-search", candidates=(2, 3, 4), seed=42, coordinates=coordinates)
other = load_experiment(Path("results/pt-search"), count=2)
other.plot(Path("results/pt-search/two-cluster-figures"), rows=1, columns=2)
```

`camels_cycles` selects precipitation and mean temperature by default. Pass an
explicit `variables` tuple to change channels, including `oudin_pet` if wanted.
PET is not added automatically. Only selected forcing is validated/read; Q is
not loaded. Coordinate rows are joined by basin ID through `read_attributes`;
coordinates are only plotting inputs. See [CAMELS preparation](camels.md).

For other sources, use `seasonal_cycles(dates, values, ids, variables, units)`.
Dates are consecutive integer Gregorian ordinals; values have axes basin, day,
variable. Metadata IDs and variable names are ordered tuples of strings. Units
are retained unchanged. An optional `provenance` dictionary records source details.
You can also provide already aggregated `SeasonalCycles` following the calendar
below. `figures=False` permits numerical experiments without coordinates; later
maps and partition profiles require coordinates in the result bundle.

## Calendar and input policy

Inputs must span complete calendar years, January 1 through December 31. API
`stop` dates are exclusive. There are 52 Jan-anchored bins: nonleap day index
(integer, zero-based) divided by seven, capped at 51. The first 51 bins have
seven days and the last has eight. February 29 enters the same bin as February
28. Daily means are calculated within each year/bin, then years receive equal
weight. Every date contributes; precipitation remains mm/day, not a weekly sum.
The annual sequence has fixed January/December endpoints, without wrapping or
circular phase invariance.

Missing/duplicate dates, incomplete years, nonfinite selected inputs, duplicate
IDs or variables, and zero pooled variance are errors. Selected CAMELS P/PET
must also be nonnegative. Diagnostics identify the first invalid basin, channel
and date. Nothing is imputed or silently excluded.

## Method and diagnostics

By default, `normalization="pooled"` uses one population mean and standard deviation
per variable over all basins and seasonal bins. Between-basin level and amplitude
differences remain. Set `normalization="per_basin"` in `run_experiment` to subtract
each basin/channel's mean and divide by its population standard deviation across
52 seasonal steps. This removes independent offsets and positive amplitude scales,
retaining seasonal shape and timing. Per-basin inputs must have no flat seasonal
curves.
Channels have equal weight after scaling. Dependent multivariate DTW uses a
single path, squared Euclidean vector local costs, and the square root of the
accumulated cost as its distance. A Sakoe–Chiba radius of **two weekly bins** is a
chosen tolerance, not an estimated optimum. It applies to initialization,
assignments, DBA and silhouettes.

The workflow uses tslearn DTW K-means with k-means++ initialization, three
restarts, 50 outer iterations and objective tolerance 1e-6 by default; `seed`,
`n_init`, `max_iter` and `tolerance` are explicit options. tslearn 0.10 uses
Petitjean DBA with 30 iterations and tolerance 1e-5. Inner DBA convergence is
**not exposed**; warnings are retained. A narrow version-constrained subclass
observes outer objective history and successful restart counts without replacing
the library algorithm. This private hook is why tslearn stays on 0.10.x.
The cluster extra pins a patched tslearn 0.10.0 Git revision that makes explicit
warping constraints apply to DBA updates as well as distances
([upstream defect](https://github.com/tslearn-team/tslearn/issues/803)). Its
`0.10.0+dtwconstraint` version is recorded in `run.json`; `pyproject.toml` and
`uv.lock` identify the exact source revision. Original unpatched CAMELS outputs
are not validated by this correction: preserve them and rerun under a new
experiment name for [#142](https://github.com/CooperBigFoot/hydrologeez/issues/142).
`converged` means the observed outer objective change met tolerance, including
on the last allowed iteration. It does not promise inner DBA convergence or a
global optimum. The selected restart's history is saved; final inertia is
recomputed from exported centers/labels (mean squared DTW), not copied from the
library's pre-update objective. Nonconverged fits remain available and flagged.
Empty-cluster failures reject the experiment rather than silently reducing k.

Silhouettes use the same constrained DTW pairwise distances, not squared costs
or a projection. Scores are undefined/null for k=1 or k=number of basins;
singletons in other fits receive zero silhouette under scikit-learn's convention.
Fixed mode fits only k and does not run automatic selection. By default,
`selection="silhouette"` search chooses the largest defined mean silhouette,
with ties resolved toward smaller k; all-undefined searches fail. This is best
separation **among tested counts**, not proof of a unique scientific optimum.
Pairwise scoring is quadratic in basin count; the 671-basin study has not been
run as implementation validation.

## Elbow selection and comparison

Use `selection="elbow"` to apply Kneedle (`kneed`) to candidate count versus
mean squared constrained-DTW inertia. The curve is fixed as convex/decreasing.
`ElbowSettings` defaults to `S=1.0`, `interp_method="interp1d"`,
`polynomial_degree=3`, and `online=False`. S is detector sensitivity, not
confidence. The default uses measured values without polynomial smoothing;
`interp_method="polynomial"` enables polynomial fitting (degree three rather
than the library default seven). `online=True` continues scanning and may revise
the elbow; otherwise the first detected elbow is returned.

Using `cycles` and `coordinates` from the example above:

```python
from hydrologeez.cluster import ElbowSettings

elbow = run_experiment(
    cycles, Path("results"), "pt-elbow", candidates=tuple(range(2, 8)),
    selection="elbow", elbow=ElbowSettings(), coordinates=coordinates,
)
original = load_experiment(elbow.directory)
silhouette = original.select(selection="silhouette")
more_sensitive = original.select(selection="elbow", elbow=ElbowSettings(S=0.5))
print(more_sensitive.selected_count, more_sensitive.selection_reason)
silhouette.plot(Path("results/pt-elbow/silhouette-figures"))
manual = load_experiment(elbow.directory, count=4)
```

Views expose `selection`, `elbow` (effective settings or None), `selected_count`,
and `selection_reason`. Reselection reuses retained fits without fitting,
mutating the original view, or writing its bundle. Supply a separate figure
directory when comparing views. Fixed `k` rejects elbow options.

An elbow suggests diminishing fit improvements; silhouette measures separation.
Neither proves a unique scientific optimum. Candidate range and detector settings
matter, even when a count is returned. No detection is a valid result with
`selected_count=None`; there is no fallback. Fewer than three candidates, flat,
nonfinite, negative, or nonmonotonic inertia curves produce explicit no-selection
diagnostics. Polynomial fitting also requires more candidates than its degree.
Nonmonotonic values are not smoothed or reordered to manufacture an elbow.

No-selection bundles retain all candidates, settings, rationale, diagnostic
values and library versions. They omit membership exports, maps and profiles;
`labels`, `centroids` and `clusters` raise a clear error. Reload, inspect candidates,
or choose a retained count manually. `.plot()` still saves silhouette and inertia
curves without needing coordinates. The inertia figure marks only a detected
elbow, including when plotting a reselected view.

## Results and figures

An existing `<output_dir>/<experiment>` is always rejected. Results include
`membership.csv`, `clusters.json`, `seasonal_cycles.npz`, `diagnostics.csv`,
`run.json` and, when requested, `figures/`. Labels are zero-based. JSON IDs are
strings; CSV readers must explicitly retain string IDs. The NPZ contains original
unwarped member curves in physical units, standardized fitted DTW barycenters
and labels for every candidate (bundle format 3). `run.json` defines all axis orders, selected count, calendar, date
range, scaling, seed, settings, warnings, outer objective histories and versions.
`load_experiment(..., count=k)` selects any retained fit without refitting or
changing the original bundle. `Experiment.normalization` records the mode;
`mean` and `std` retain its scaling parameters. `normalized_members` reconstructs
the fitted inputs. `Experiment.centroids` is in physical units for pooled runs
and standardized, dimensionless units for per-basin runs; `centroid_units` labels
these explicitly. Internal `Candidate.centroids` is always scaled. Per-basin
centroids have no unique physical-unit inverse. Existing format-1 and format-2 bundles remain reloadable.

Figures use a consistent six-color palette extended with golden-angle hues for
larger k. Every cluster map and profile uses the same color. Grey profiles are
actual member climatologies; colored curves are DTW barycenters, not arithmetic
averages. Each variable has its own profile figure, in both normalization modes.
Pooled profiles use physical units; per-basin profiles use standardized,
dimensionless member curves and centroids. Scales are common within a variable.
Pass `rows=2, columns=2` or `rows=1, columns=4` to `run_experiment` or `.plot()`
to choose the cluster grid. An omitted dimension is inferred; with both omitted,
the grid uses at most two columns. Undersized grids are rejected and unused panels
are hidden. The former `layout` argument has been removed: omit it for separate
figures and use `rows`/`columns` to control their arrangement. Combined-variable
figures are no longer generated. Search produces silhouette and inertia diagnostics. The map is
separate, with a CONUS extent; use CONUS coordinates for this workflow.
Cartopy downloads Natural Earth boundaries on first use; offline plotting needs
its populated data cache. A plotting failure leaves the fitted numerical bundle
available for `load_experiment(...).plot()` rather than requiring another fit.
