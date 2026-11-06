# Constrained DTW centroid correction

Make DTW centroid updates obey the same warping constraint as assignment and scoring in hydrologeez’s seasonal clustering workflow. Implement the correction in tslearn, backport it to 0.10.0, and use that patched dependency reproducibly from hydrologeez. Do not replace the explicit constraint configuration with the radius-only workaround.

This vision delivers the correction and dependency-integration validation tracked by [#149](https://github.com/CooperBigFoot/hydrologeez/issues/149). It does not complete the full CAMELS rerun: that remains under [#142](https://github.com/CooperBigFoot/hydrologeez/issues/142). The original run’s memberships and selected cluster count remain unvalidated until recomputed with the correction. Preserve the original outputs rather than overwriting them.

## Confirmed defect and intended behavior

`src/hydrologeez/cluster/fit.py` passes `global_constraint="sakoe_chiba"` and `sakoe_chiba_radius` to `TimeSeriesKMeans`. Assignment and silhouette distances use the intended constraint. DBA centroid updates forward the string to a private DTW kernel that expects an integer constraint code; the mask therefore becomes unrestricted.

The minimal reproduction in [upstream tslearn #803](https://github.com/tslearn-team/tslearn/issues/803) fails on installed tslearn 0.10.0 and on the local source checkout at `c8711aa7faf4bd2e68726b8b3fafa2df0f2d284d` (`0.11.0.dev`). Two equal-length series, `[0, 0, 1, 0, 0]` and `[0, 1, 0, 0, 0]`, with an explicit Sakoe–Chiba radius of zero must yield their pointwise mean, `[0, 0.5, 0.5, 0, 0]`. Initialized from the first series, the current Petitjean implementation instead returns that first series unchanged. Passing only the radius works for this reproduction, but the agreed approach is to repair the library’s explicit-constraint API.

Source inspection found the same conversion bypass in `_petitjean_assignment` and `_mm_assignment`, in serial and parallel paths. These helpers serve Petitjean, MM and subgradient DBA. The public `dtw_path` already uses `GLOBAL_CONSTRAINT_CODE` to translate named constraints. Correct this boundary consistently, preserving public parameter semantics and avoiding mutation of the caller’s parameter dictionary. Leave the precise code structure to implementation; do not add a general compatibility layer or fork the clustering algorithm into hydrologeez.

## Dependency and upstream delivery

The existing local tslearn clone is `/Users/nicolaslazaro/Desktop/thirdparty/tslearn`. It currently contains development version `0.11.0.dev`, while hydrologeez requires `tslearn>=0.10.0,<0.11`. Prepare the library correction so it can be contributed upstream, and backport it onto 0.10.0 for hydrologeez. Do not combine this fix with a dependency upgrade.

Use an identifiable, immutable patched revision or artifact that another checkout can resolve. A developer’s absolute local path alone is not a reproducible dependency. Leave the exact pinning and packaging mechanism to the implementing agent. Keep dependency provenance clear enough to identify the patched source used for later scientific runs; use existing metadata where sufficient rather than introducing a new provenance subsystem.

The bug report is already published as tslearn #803, using the project’s contribution guidance and bug-report template. It offers a tested fix PR if the maintainers are interested. Do not open an upstream PR unless the team expresses interest. Local correction and hydrologeez integration need not wait for that response. Follow tslearn’s contribution and testing guidance for library changes.

## Evidence of completion

Develop the fix test-first. Observe the explicit-constraint reproduction failing for the intended reason, then passing after the library change. Test actual barycenter values, not just forwarded parameters or mocked distance calls. Cover the affected assignment helpers and serial/parallel execution with small, independently understandable cases. Preserve unconstrained behavior and caller parameters; cover distinct named-constraint behavior where relevant without building an exhaustive test matrix. Subgradient tests must respect its iterative behavior rather than assuming one stochastic update reaches the exact mean.

In hydrologeez, add a focused integration regression proving that the installed patched dependency makes `TimeSeriesKMeans` centroid updates honor the explicit constraint. Retain the existing shared multivariate alignment, pooled channel scaling, configured two-bin scientific tolerance, and constrained assignment/scoring contract from the [seasonal clustering vision](2026-11-05-seasonal-time-series-clustering-experiments.md). Radius zero is a test discriminator, not a change to the scientific tolerance.

Verify dependency resolution and the relevant tslearn and hydrologeez tests in their project environments. Use uv for hydrologeez execution, strong typing for its changes, and its Ruff and ty checks. Record the patch revision and validation results. No full 671-basin search, plotting redesign, clustering-method change, or claim about the corrected optimal cluster count belongs in this delivery.

Implementation is complete when the patched 0.10.0 dependency is reproducibly usable by hydrologeez and behavioral tests establish consistent constrained centroid updates, assignment and scoring. Publication of this vision alone does not complete #149. Report the corrected workflow as ready for the separate #142 rerun, not as validation of the old results.
