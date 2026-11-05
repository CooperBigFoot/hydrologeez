# Seasonal time-series clustering experiments

Implement a Python workflow for grouping basins by seasonal climate regime, initially using CAMELS precipitation (P) and mean temperature (T). The user runs named experiments with different variable selections or cluster counts, obtains basin-ID lists for downstream analysis, and inspects publication-style figures. The core belongs under `src/hydrologeez/cluster`; keep CAMELS loading separate from general clustering of selected time-series channels.

This vision covers implementation, not the full CAMELS analysis. [#142](https://github.com/CooperBigFoot/hydrologeez/issues/142) tracks running and inspecting that analysis once this workflow exists. Attribute-based interpretation/classification and assigning new basins from attributes are deferred to [#141](https://github.com/CooperBigFoot/hydrologeez/issues/141). Neither issue is completed by this vision or its implementation.

## Scientific meaning

Cluster seasonal climate regimes, not coincident multi-year weather events or inferred hydrological response. Build weekly seasonal climatologies by averaging corresponding seasonal periods across years. For the subsequent CAMELS analysis, use the available 1980–2014 records for all 671 basins, initially P and T. Do not discard basins merely because their Q observations are incomplete.

P and T remain separate channels: each seasonal step is a vector, not a scalar PC1 projection. Permit other selected time-series variables without hard-coding a P/T-only distance. Standardize each variable using one pooled mean and standard deviation across basins and seasonal steps. Do not standardize each basin separately: between-basin wetness, warmth and amplitude differences must remain meaningful. Use equal channel weighting after scaling initially; learned or user-designed weighting is not required.

Use dependent multivariate Dynamic Time Warping (DTW): one alignment path shared by all channels, with squared Euclidean local costs on scaled vectors. Constrain alignment to a maximum shift of two weekly steps. This tolerance is a user-selected assumption, not an empirically established optimum. Use DTW-based K-means with multivariate DTW barycenter centroids, preserving one centroid curve per variable. PCA compression, independently warped channels, and unrestricted warping are not the agreed method.

Calendar binning, leap-day handling, and treatment of the annual boundary must be explicit, deterministic, and documented. Leave their exact implementation to the implementing agent; preserve seasonal order and do not silently drop dates or introduce arbitrary circular phase invariance. Weekly climatologies are not a concatenation of all weeks across all years. Invalid or incomplete requested inputs must receive an explicit policy and diagnostics rather than silent imputation or exclusion.

## Experiments and results

Users supply an output directory and an experiment name. Store the complete result bundle under `<output_dir>/<experiment>/`. Reject an existing experiment folder rather than silently overwriting it. Provide a Python result usable directly by downstream code as well as persisted outputs; exact function signatures and whether to add a thin CLI are implementation choices.

Support two explicit modes:

- Fixed cluster count: fit the requested `k` without a cluster-count sweep; still calculate its silhouette score when defined.
- Search: the user supplies candidate counts. Fit and score each, select the highest mean silhouette, and retain diagnostics so the user can inspect the choice. Do not invent an implicit search range. State that the peak is the best separation among tested counts under the selected distance, not proof of a unique scientific optimum.

Compute silhouettes with the same constrained multivariate DTW distance used for assignments, not a Euclidean proxy or a two-dimensional projection. Make randomness reproducible through a recorded seed. Library selection, initialization/restarts, convergence settings, and an explicit policy for undefined silhouette cases remain ordinary implementation decisions; use established Python implementations where suitable rather than porting the hand-written Rust machinery wholesale.

The agreed result bundle is:

```text
<output_dir>/<experiment>/
  membership.csv       # basin_id, cluster
  clusters.json        # cluster -> list of string basin IDs
  seasonal_cycles.npz  # member seasonal curves and centroid curves in original units
  diagnostics.csv      # cluster-count scores and fit diagnostics
  run.json             # configuration, variable/axis metadata, scaling, seed and provenance
  figures/             # PNG and SVG outputs
```

Keep basin IDs as strings, including leading zeros. Preserve unambiguous basin, variable, cluster and seasonal-step ordering in the persisted metadata. Store enough information to redraw the figures without fitting again, including selected count, scaling parameters, time range and calendar policy. For search mode retain fitted candidate results sufficiently to select a previously evaluated count without repeating the search; the internal storage arrangement is an implementation choice. Record actual convergence and any exclusions rather than implying all fits succeeded.

## Scientific figures

Use a consistent, deliberately chosen cluster palette across the map and centroid curves, not the default categorical cycle. Follow `sns.set_context("paper", font_scale=1.3)` and `sns.despine()` where applicable. Use black labels/ticks, white backgrounds, no figure titles or explanatory footers, and only essential units, cluster identifiers, counts and legends. Base single figures on 8 × 6 inches and adjust grids proportionately. Retain seasonal ticks but omit time-axis titles.

Deliver:

1. Mean silhouette versus candidate cluster count for searches. No fabricated uncertainty band. Fixed-count experiments must not imply a search was performed.
2. A separate Cartopy map of the contiguous US with all included CAMELS gauge locations coloured by cluster and a clear legend. Coordinates are plotting inputs, not clustering predictors.
3. Seasonal curves with one panel per cluster and variable: every member as a thin, light-grey curve, and the centroid as a solid line in the map's cluster colour. Keep P and T on separate axes, never dual-axis overlays or precipitation bars. Use original units (P in mm/day, T in °C), with common scales within each variable. Support the approved paired layout (cluster rows, variable columns) and separate-variable grids. Do not embed the map in these figures.

The coloured line is the inverse-scaled DTW barycenter, not a pointwise arithmetic average. Member curves remain their unwarped seasonal climatologies so viewers can see actual timing and spread.

Approved visual mockups are available locally in `~/Desktop/camels-clustering-mockups`: `01_cluster_count_diagnostic`, `02_conus_cluster_map`, `03_seasonal_profiles_by_cluster`, `04_precipitation_profiles`, and `05_temperature_profiles`, with editable rendering code. They are visual references only: assignments, scores and member curves are synthetic, and their coloured representative lines are arithmetic means, not fitted DTW barycenters. The initial six-colour palette is teal `#287A78`, ochre `#C17A32`, violet `#766498`, rose `#B85362`, olive `#648447`, and blue `#397FA4`; handle other cluster counts deliberately. The implementation must not depend on these local mockup files.

## Repository context and boundaries

`hydrologeez.camels.Camels` reads prepared daily inputs and supports ordered basin/date selections; see [CAMELS documentation](../../docs/camels.md). Prepared forcing has P, mean T and calculated Oudin PET, and observations include Q with missingness. Oudin PET depends on T and latitude/season, so adding it is an explicit variable choice, not an automatic extra channel. Preserve compatibility with the existing reader without tying the clustering math to it.

Static attributes are already prepared independently after PRs #139 and #140. `read_attributes(...)` returns `CamelsAttributes` with a resident Polars `.table`, string `gauge_id`, and `.metadata`. Join coordinates for maps by basin ID. Do not rebuild static preparation or include attributes in clustering. The classifier remains separate work.

The neighbouring `narcissus` repository is conceptual context: optional per-basin normalization, scalar DTW K-means and DTW barycenter averaging, with basin labels and centroids. It contains neither PCA reduction nor a seasonal-climatology builder or plotting pipeline. This vision intentionally differs in pooled normalization and multivariate input. The supplied He et al. (2024), DOI 10.1016/j.jhydrol.2024.131638, clusters multi-year weekly Q, not weekly-of-year climatologies; this work is not an exact replication. No random forest, SHAP workflow, automatic attribute selection, or classifier evaluation belongs in this implementation.

## Evidence of completion

Use the repository's uv environment, strong typing, Ruff, ty and test-first guidance. Prove with small independently understandable inputs that seasonal aggregation follows the documented calendar, pooled scaling retains between-basin differences, both channels influence a shared constrained DTW alignment, and basin identities survive selection and output round trips. Cover fixed-count versus search behavior, silhouette-based selection, reproducibility, and collision rejection. Verify stored member/centroid curves and metadata are sufficient for plotting in original units, and render/inspect the scientific figures without asserting incidental styling strings.

The workflow must be usable for a small named experiment and an explicit output location without first executing the full CAMELS study. Record practical limitations rather than claiming a clustering optimum or scientific conclusion from implementation tests. The actual 671-basin analysis and its chosen cluster count belong to #142.
