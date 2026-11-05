# Static catchment attribute clustering

Group catchments from explicitly selected numeric static attributes and display
the groups on a map. Provide a usable Python workflow with reproducible named
experiments, similar to the existing seasonal DTW clustering workflow. The
priority is basin grouping and mapping, not scientific analysis or attribute
interpretation figures.

Implementation completes [#160](https://github.com/CooperBigFoot/hydrologeez/issues/160).
Publication does not complete the issue. This confirmed scope supersedes the
issue's earlier request for attribute profiles and summaries: neither is required.

## Inputs and clustering

- Consume prepared attributes exposed by `read_attributes(...)` and
  `CamelsAttributes.table`. Users explicitly select predictor fields; never
  silently select all columns or default to the EA-LSTM attribute list.
- Support numeric predictors only. Reject selected categorical fields. Basin
  IDs identify rows and remain strings, including leading zeros. Map coordinates
  are separate plotting inputs, not clustering predictors.
- Fail on missing or nonfinite values in selected predictors, with diagnostics
  identifying affected fields and catchments. Do not impute or exclude catchments.
  Missing values in unselected fields do not prevent a run.
- Standardize each selected attribute across the selected catchments by its mean
  and standard deviation, then fit ordinary Euclidean K-means. Selected features
  have equal weight after scaling; no feature weighting or algorithm menu is
  required. Reject zero-variance selected fields clearly rather than silently
  dropping them. Preserve fitted scaling parameters.
- Keep basin identities aligned through fitting, exported memberships and maps.
  No time-series inputs or existing cluster labels are required.

Support fixed cluster counts and explicit candidate-count searches with silhouette
or elbow selection, following the existing workflow's semantics. Fixed-count runs
do not search; searches retain candidate fits and diagnostics for inspection and
manual selection. Use Euclidean silhouettes and K-means inertia, not DTW distances.
Preserve the existing distinction between a selected count and an elbow search
that finds no elbow; do not invent a fallback partition. Reuse suitable existing
selection behavior without forcing static vectors through seasonal representations.

## Results and map

Save reproducible named experiments containing basin memberships, selected fields,
fitted scaling, algorithm settings, seed, count-selection settings and diagnostics.
Retain the inputs and candidate results needed for reload, inspection and mapping
without rereading source data or refitting. Follow existing conventions for rejecting
an existing experiment directory, choosing retained counts and keeping numerical
results available if plotting fails.

Produce a cluster map using the existing DTW map styling and CONUS scope. Coordinates
must align by basin identity. Keep count-selection diagnostics available, including
the existing silhouette/inertia diagnostic plots where applicable. An unselected
search retains diagnostics but has no selected membership or cluster map until a
count is chosen.

Retain and expose each cluster's K-means centroid: the arithmetic mean of its
members in standardized attribute space. Also expose its inverse-transformed
attribute vector in original units, with the selected field order preserved.
These representative vectors are the static-feature counterpart of the DTW
barycenters, not DTW barycenters themselves. Persist centroids for retained
candidate fits and preserve them through reload and count selection.

Do not produce attribute profile figures or additional interpretive attribute-summary
tables. The centroid vectors above are required model outputs, not excluded summaries.
No classifier, new-basin prediction interface, scientific study, automatic attribute
selection, additional clustering methods or generic clustering framework is requested.

## Repository context and implementation latitude

The existing workflow is documented in [cluster.md](../../docs/cluster.md), with
implementation under `src/hydrologeez/cluster/`. Its result and input types are
seasonal-specific today. Preserve existing seasonal functionality; reuse small
shared pieces where appropriate rather than broadening every seasonal abstraction.
Exact API names, result types and storage layout are implementation choices.

[CAMELS preparation](../../docs/camels.md#static-attributes) describes the separate
static-attribute store and labeled Polars table. Daily prepared records alone are
not that store. Local source inspection found that all 27 numeric attributes used
by the [original EA-LSTM implementation](https://github.com/kratzert/ealstm_regional_modeling/blob/d11815824793737f28f125d55477e42f8900a78e/papercode/datautils.py)
are complete across our 671 CAMELS catchments. Other fields contain missing values;
validation must depend on the selected fields. The EA-LSTM list is an optional
user selection, not a new preset or implicit contract.

Use project-native uv commands, strongly typed Python, Ruff, ty and focused
test-first development. Demonstrate numeric selection and scaling, meaningful
K-means grouping, identity alignment, rejection of invalid selected inputs,
fixed/search modes, reloadable memberships and settings, and map generation.
Reuse existing selection tests where possible. Document a small runnable example
using explicit attributes; no full CAMELS clustering study is needed. No existing
scientific constants or DTW settings should change.
