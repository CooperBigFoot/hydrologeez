# Persistent streamflow QC and experiment preprocessing

Implement reusable scientific quality control and experiment preprocessing in hydrologeez. The user should be able to choose data X and processing Y, prepare it once, store it on disk, and train repeatedly without rebuilding the workflow or fitting transformations in the loader.

This vision develops the data workflow discussed in [#192](https://github.com/CooperBigFoot/hydrologeez/issues/192). Implementing it does **not** complete #192: discharge output-head, differentiable transformed-loss, ensemble inversion and checkpoint contracts still need focused design. Publication delivers no implementation.

## Two persistent stages

**Source data → retrospective QC archive → experiment selection and preprocessing → prepared dataset on disk → runtime batches.**

The QC archive retains observations and assessments; experiments decide which observations and basins to exclude. Neither storage layout nor model requirements should silently determine scientific eligibility. “QC-controlled” means screened under recorded rules, not guaranteed correct.

### Retrospective streamflow QC

Implement all three parts of [Gudmundsson et al. (2018), GSIM Part 2](https://doi.org/10.5194/essd-10-787-2018):

- Daily discharge screening, including the conditional routing in sections 2.2–2.3: use reliable provider flags where available and the automated checks where reliable flags are unavailable or their use is discouraged. Do not silently run both branches and combine their exclusions; preserve the paper's treatment of partially flagged records.
- Monthly, seasonal and annual streamflow indices, with their data-availability criteria.
- Homogeneity assessment of those indices, including the paper's preprocessing, tests and classification.

All three capabilities are in scope; a caller need not apply all three to every dataset. Use retrospective full-record screening, not training-cutoff-specific QC. Document that future observations can influence earlier statistical flags. This is distinct from the training-only fitting rules below.

Preserve original observations, source missingness and available provider flags. Persist separate daily QC reasons, index reliability and homogeneity assessments, with sufficient units, dates, basin identities, method configuration and source provenance to interpret and reproduce them. Do not discard basins, overwrite original values, repair observations or automatically enforce scientific exclusions in the archive. Calculations of indices may exclude suspect daily values as prescribed by the method without deleting those values from the archive. Homogeneity findings are assessments, not proof of measurement error: real regulation can change a river's behavior.

Use the paper as the scientific reference, not the old implementation. Its automated daily checks include negative flow, more than 10 consecutive identical positive daily values, and seasonal outliers in `log(Q + 0.01)` outside six standard deviations. The latter uses a five-calendar-day window pooled across years. Preserve the method's unit convention: the additive offset is not invariant under conversion between m³/s and mm/day. Do not substitute `log1p(Q)` or add a long-zero-run rejection rule and call it the published method.

Resolve numerical details against the paper and cited methods, documenting necessary conventions where the sources are silent. Handle date ordering, gaps, leap calendars, missing/nonfinite values, insufficient data and constant series explicitly. Calendar windows mean calendar days, not arbitrary adjacent rows. Scientific results must be independent of file/chunk/reader batch boundaries. The homogeneity procedure must include the prescribed detrending, seasonal treatment and pre-whitening rather than simply applying tests to raw daily discharge.

### Experiment preprocessing

Let callers choose basin/date/feature selections, exclusion rules based on stored observation flags or basin/index assessments, authorized fitting pools, transform chains and global or per-basin scope. Keep scientific campaign rules caller-owned rather than embedding a particular dataset or split in core utilities.

Support log transforms and z-scoring, with explicit log configuration, units and feature order. Apply chains in the declared order and provide inversion where mathematically defined. Global means a shared fitted parameter set per feature across the authorized pool; per-basin means separate fitted parameters by basin. Static-field statistics count each authorized basin once, not once per daily record or training sample. Preserve missingness and masks through transformation.

Fit statistics only on accepted, observed values in explicitly authorized pools; never on rejected observations, filled replacements or implicitly accessed validation/test values. Persist fitting scope and parameters and never refit on loading. An unseen basin cannot silently acquire a per-basin transform from its evaluation targets; unsupported fitting/application combinations must be explicit. Discharge transforms remain supervised-training-only and fold-specific. Other inputs may use explicitly authorized unlabeled pools when an experiment calls for them, as described in #192.

After transforming accepted input values, replace missing inputs with a configurable **constant in final model-input space**, default **0**. This includes observations made unavailable by the experiment's exclusion policy. Support no interpolation or forward-fill in this work. Produce a per-field `_was_filled` flag wherever a replacement actually occurs, retain the underlying missingness/QC evidence, and leave flags untransformed. Expose these flags for use as model inputs; they are not QC verdicts.

**Filled targets never contribute to training loss or evaluation.** A finite placeholder does not make a target observed or eligible. Carry an explicit target-eligibility mask to consumers; preserve the distinction between filled predictors and observed targets. Statistics are fitted before filling, so placeholder constants cannot influence them.

Persist the actual prepared values and flags on disk, together with the recipe, fitted transform parameters, selected identities/order, units and provenance. This is not merely saving a recipe for transformations to be repeated during loading. Repeated model runs consume the prepared artifact without repeating preprocessing. Keep the QC archive available so another experiment can choose different exclusions and preprocessing.

## Integration and scope boundaries

Physical layout means how data are stored on disk; runtime representation means how they are modeled in memory. Reuse the existing efficient prepared-data reader and resident runtime where suitable. Keep QC independent of physical I/O batching and avoid fitting or scientific policy inside the hot loader path. `src/hydrologeez/quality_control/` is a reasonable module home, not a mandate for a new framework.

Relevant current integration points include `camels.py`, `camels_meteorology.py`, `hybrid/camels.py`, `hybrid/data.py`, and the LSTM input adapter. Existing hybrid inputs load an authorized pool once and slice resident tensors. Existing meteorological normalization already fits explicitly across basin/day axes. Extend or consolidate suitable functionality rather than creating duplicate normalization systems.

Keep physical precipitation, temperature and PET available unchanged for HBV. Filling neural inputs is not authorization to drive HBV with standardized placeholders or invalid physical forcing. Preserve physical discharge and the distinction between physical and transformed values. This vision does not settle standalone DL output-head policy or the rest of #192's transformed-loss contract.

The legacy workflow provides context, not compatibility requirements: Metis prepared experiment versions, Hydria loaded them, and Nostos staged training. Retain useful separation and provenance without importing the multi-project stack, HDX requirements or its incidental restrictions. Inspection found QC in `caravan-cli/src/qc/` with batch-local statistics, constant-run boundary errors and different log semantics; do not port those behaviors. Older ingestion also discarded some provider reasons or excluded basins before storage, which conflicts with experiment-owned exclusions here.

Implement logic, persistence and reader/runtime integration—not a Hetzner QC run, dataset migration, campaign execution or model training. Do not provision infrastructure or change the large stored dataset. No automatic gap repair beyond the agreed constant input filling, new QC heuristics, mandatory basin rejection, new storage engine or general workflow framework is requested. Exact APIs and storage grouping remain implementation choices. If evidence calls for a substantive scientific departure from the paper, raise it rather than silently adopting it.

## Evidence of completion

Use focused, independently anchored tests and small fixtures to demonstrate:

- Published daily checks, representative indices and homogeneity outcomes, including reliability/insufficient-data states and documented numerical conventions.
- Identical QC results across different storage/reader batch boundaries and correct calendar/run behavior.
- Archive round-tripping of original values, reasons and assessments; experiment exclusions do not destroy the archive.
- Correct global/per-basin fitting, static weighting and authorized-pool isolation; explicit invalid-input and constant-feature behavior; transform ordering, inversion and persistence.
- Constant filling after transformation, exact `_was_filled` provenance, unchanged QC evidence, no synthetic contribution to fitted statistics, and exclusion of filled targets from loss/evaluation.
- Prepared artifact reload and repeated consumption without refitting or repeating transformations, with ordered data/masks aligned and physical HBV inputs preserved.

Provide concise usage showing the two persisted stages and model consumption. Use project-native `uv`, strong Python typing, Ruff, ty and test-first development. Tests and bounded local fixtures are authorized; processing the user's large dataset is not.
