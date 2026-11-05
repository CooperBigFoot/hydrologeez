# Selectable cluster-count selection

Add elbow-based automatic cluster-count selection alongside silhouette selection in the seasonal clustering workflow. The interface must be ergonomic for agents: explicit method names, typed settings, inspectable outcomes, and reuse of retained fits without refitting or parsing prose to obtain the selected count.

Implementation of this vision completes [#157](https://github.com/CooperBigFoot/hydrologeez/issues/157). Publishing this document does not. This is tooling only: CAMELS execution and scientific interpretation remain in [#142](https://github.com/CooperBigFoot/hydrologeez/issues/142). Do not change the scientific partition or run a new CAMELS study as part of this work. Earlier repeated silhouette selections of two clusters motivate comparison, not an assumption that two is wrong; current documentation also requires corrected DTW runs before relying on earlier CAMELS outputs.

## Selection and API

Preserve `run_experiment` and add `selection="silhouette" | "elbow"`, defaulting to `"silhouette"` for searches. Keep the explicit candidate counts and existing fixed-count `k` mode. Fixed-count runs do not perform automatic selection. Preserve silhouette semantics: highest defined mean silhouette, ties toward smaller counts, and failure when all search scores are undefined.

Expose a typed `ElbowSettings` through the cluster package, passed using `elbow=`. Use Kneedle through `kneed`, not a new custom detector. Settings and defaults:

- `S=1.0`: the detector sensitivity threshold, not a probability or confidence level.
- `interp_method="interp1d"`: use the measured values without polynomial smoothing; optionally accept `"polynomial"`.
- `polynomial_degree=3`: relevant only to polynomial fitting. This deliberately differs from the library default of seven, which is high for a short candidate sweep.
- `online=False`: return the first detected elbow; allow `True` to continue scanning and revise the detected elbow according to library semantics.

Fix `curve="convex"` and `direction="decreasing"` internally: these describe the intended inertia curve, not useful user tuning choices. Validate settings and incompatible argument combinations clearly; exact validation organization is an implementation detail.

Illustrative calls (existing coordinate/figure options omitted):

```python
result = run_experiment(
    cycles, output_dir, "silhouette-search",
    candidates=tuple(range(2, 11)), selection="silhouette",
)
result = run_experiment(
    cycles, output_dir, "elbow-search",
    candidates=tuple(range(2, 11)), selection="elbow",
    elbow=ElbowSettings(S=1.0),
)
fixed = run_experiment(cycles, output_dir, "fixed-four", k=4)

original = load_experiment(output_dir / "elbow-search")
silhouette = original.select(selection="silhouette")
elbow = original.select(selection="elbow", elbow=ElbowSettings(S=2.0))
manual = load_experiment(output_dir / "elbow-search", count=4)
```

`Experiment.select()` returns a new result view using retained candidate fits; it does not refit, mutate the original view, or overwrite its bundle. Each view carries its method, settings, selected count, and rationale. Keep the existing manual `load_experiment(..., count=k)` operation. No separate experiment API or CLI is required.

## Scientific meaning and no-selection behavior

Apply the detector to candidate count versus existing inertia: mean squared constrained-DTW distance to assigned centroids, recomputed from the exported fitted centers and labels. Do not use a Euclidean proxy. Preserve both inertia and silhouette diagnostics for method comparison.

An elbow suggests diminishing improvements in within-group fit; silhouette measures separation. Neither identifies a unique scientific optimum. The candidate range and detector settings affect the answer. A detector returning a count is not proof of an unambiguous scientific elbow.

“No elbow detected with these settings” is a valid, reloadable outcome. Represent it as `selected_count=None`, with `selection_reason` explaining the outcome and all `candidates` retained. Do not fall back to silhouette or an arbitrary count. Flat, insufficient, or otherwise unusable curves must yield an explicit diagnostic rather than an invented selection; leave the precise checks to implementation and document them. Do not silently alter noisy/nonmonotonic fitted inertia to manufacture a convincing elbow.

With no selection, save and allow plotting the diagnostic curves, but do not generate selected membership exports, maps, or centroid-profile figures. Access to properties requiring a selected partition must raise a clear error. Users can still inspect retained candidates or explicitly select a retained count. This behavior must work through initial execution, reload, and reselection.

## Persistence and figures

Save the selection method, effective detector settings, diagnostic curve, selected count or null, rationale, and relevant library version in reloadable results. Preserve the original bundle when comparing methods through result views. Initial run selections must round-trip; exact serialization details and any additional explicit persistence mechanism for alternate views are left to implementation, not an invitation to add a new export subsystem.

Provide an inertia-versus-count figure marking the detected elbow when present; no marker should imply a selection when none exists. Keep silhouette diagnostics and plotting available, including when comparing methods. Follow existing figure conventions. Plotting a reselected view must reflect that view's selection, not stale bundle metadata.

## Repository context and verification

The relevant implementation is under `src/hydrologeez/cluster/`, particularly `experiment.py`, `fit.py`, and `plotting.py`; user documentation is `docs/cluster.md`. Current candidates already retain inertia, silhouettes, labels, and centroids. Current experiment serialization and selected-partition properties assume an integer selected count, so nullable selection needs to be handled throughout, not only inside the detector. Preserve loading of existing supported result bundles and their silhouette behavior.

Use project-native `uv` commands, strongly typed Python, Ruff, ty, and focused test-first development. Demonstrate a clear elbow and an ambiguous/no-detection curve with independently justified expectations; verify settings, default silhouette behavior, fixed-count behavior, retained-fit comparison without refitting, persistence, and diagnostic plotting with and without a selection. Reuse existing coverage rather than duplicating it. Document the detector, defaults, range dependence, and explicit no-detection behavior with a short runnable API example.

Implementation owns reversible internal choices such as serialization layout, validation helpers, and precise result metadata representation. Do not add confidence scores, automatic range expansion, scientific partition changes, or a broader model-selection framework.
