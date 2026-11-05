# Per-basin normalization and separate variable profiles

Add an optional per-basin normalization mode to seasonal clustering so users can focus on seasonal shape. Make profile figures separate by variable in **both** normalization modes, with configurable cluster grids. Implementing this vision completes [#154](https://github.com/CooperBigFoot/hydrologeez/issues/154); publishing it does not.

This extends the [seasonal clustering vision](2026-11-05-seasonal-time-series-clustering-experiments.md). It supersedes that document's pooled-only normalization and combined-variable profile requirements. Other clustering behavior remains unchanged. Do not rerun or replace the CAMELS experiment tracked in [#142](https://github.com/CooperBigFoot/hydrologeez/issues/142), select a new scientific partition, or expand into a scientific-method investigation.

## Normalization and results

Keep pooled scaling as the default and preserve its existing numerical behavior. Add an explicit per-basin option: for each basin and selected variable independently, subtract the mean across its 52 seasonal steps and divide by the standard deviation across those steps. This removes level and amplitude differences while retaining seasonal shape and timing before the existing constrained DTW fit. Do not change channel selection, equal channel weighting after scaling, shared DTW alignment, or the two-week shift allowance.

Inputs are assumed to have no flat seasonal curves. No special flat-curve workflow, basin exclusion policy, or scientific threshold for weak seasonality is required.

Record the normalization mode and scaling parameters in reloadable results. Retain physical-unit member climatologies. Per-basin runs must retain normalized fitted centroids and enough information to reconstruct normalized member curves, including when selecting any previously evaluated cluster count without refitting. A per-basin normalized centroid has no unique physical-unit inverse: do not invent one or apply one basin's scale to a cluster centroid. Pooled results continue to support their existing physical-unit centroids and profiles. Keep existing pooled result bundles reloadable.

## Figures

Profile layout is independent of normalization. Produce one profile figure per selected variable, each containing its cluster panels; do not produce a combined-variable profile figure. This applies to both fresh runs and replotting saved results.

Allow explicit grid rows and columns, such as 2×2 or 1×4, with enough panels for all selected clusters. Reject undersized grids and hide unused panels. Choose a sensible automatic grid when dimensions are omitted; exact API and dimension inference are implementation details.

For pooled runs, plot members and fitted centroids in physical units. For per-basin runs, plot normalized members and fitted centroids with labels that clearly identify standardized, dimensionless values. No additional physical-unit profile family is requested for per-basin runs. Preserve thin grey member curves, cluster-coloured centroids, consistent map/profile colours, common scales within each variable, and the existing minimal styling.

Remove the black outer frame from the CONUS map and centre its legend at the bottom, below the map. Apply these changes in both normalization modes.

## Implementation context and evidence

The relevant code is under `src/hydrologeez/cluster/`:

- `fit.py` provides `pooled_scale`; `experiment.py` currently always calls it.
- `Experiment.centroids`, persistence, loading, and plotting currently assume a single invertible pooled transform. Adapt these together so centroid units remain unambiguous.
- `plotting.py` already has a separate-variable renderer, but fixes it to at most two columns. Use that existing path rather than building a second plotting system.

Use the repository's uv environment, strong typing, Ruff, ty, and test-first guidance. Demonstrate that per-basin normalization removes independent basin/channel offsets and positive rescaling, and yields the expected standardized curves. Preserve pooled reproducibility. Verify saved-mode/scaling round trips, selection of saved candidate counts, and replotting without fitting. Render and inspect separate-variable figures for both modes, explicit grids, and the revised map. Keep tests focused on behavior rather than incidental styling strings.

Document the normalization option, figure units, and separate-variable grid controls in the existing clustering documentation. Public signatures and storage mechanics remain implementation choices; keep the interface small.
