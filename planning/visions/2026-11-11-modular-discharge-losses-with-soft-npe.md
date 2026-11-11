# Modular discharge losses with soft NPE

Make MSE and a differentiable approximation of nonparametric Kling–Gupta efficiency available as reusable discharge losses, and allow the hybrid HBV trainer to select between them. The same losses should be usable by a network that predicts discharge directly, without depending on HBV or its training runtime.

This vision completes [#181](https://github.com/CooperBigFoot/hydrologeez/issues/181) when implemented and verified. Publishing this document does not complete the issue. [#182](https://github.com/CooperBigFoot/hydrologeez/issues/182), exploring soft ranks as an improvement to the evaluation metric itself, is separate and remains out of scope.

## Model and repository context

The hybrid model predicts HBV parameters, passes them through differentiable HBV, and learns from the resulting discharge error. No ground-truth parameter labels are needed. Changing the discharge loss must preserve that gradient path, including when the encoder is frozen and only the parameter head trains.

Currently, `hybrid/runtime.py` hardcodes masked discharge MSE. `masked_mse` lives in `hybrid/data.py` and also serves meteorological reconstruction pretraining. Here, masking means selecting scored observations, not hiding meteorological inputs. `metrics.py` supplies exact NumPy/SciPy evaluation metrics and calibration objective conversions, including `kge_np`. The normalizing flow's negative parameter log likelihood in `flow/runtime.py` addresses a different task.

## Scope and module boundaries

- Introduce a small shared `hydrologeez.losses` module for differentiable MSE and soft NPE. Prefer plain functions and the smallest useful selection/configuration surface; no registry, plugin framework, or class hierarchy without a demonstrated need.
- Losses consume predicted discharge, observed discharge, and observation validity, not models or training batches. Make tensor dimensions and reduction semantics explicit.
- Training owns model execution, optimizer updates, and selection of scored periods, including warmup exclusion. Losses own their mathematical calculations over selected observations.
- Move shared numerical logic rather than duplicate it. Preserve reconstruction pretraining behavior and existing MSE behavior by default. Do not apply soft NPE to meteorological reconstruction.
- Keep exact evaluation and traditional calibration behavior in `metrics.py` unchanged. Clearly distinguish the approximate training loss in names and documentation, for example `soft_npe_loss`.
- Preserve the selected discharge loss and relevant settings through checkpoint save/load so resumed training uses the same objective. Decide compatibility details in light of the existing checkpoint contract.
- Do not change HBV structure, parameter bounds, encoder architecture, or the flow model's likelihood objective. Do not add other losses or implement a new direct-discharge model.

Modularity and architectural fit are part of the outcome, not an excuse for a broad refactor. Inspect current callers and tests before choosing interfaces; keep changes focused on shared losses and their integration.

## Soft NPE

NPE here means the nonparametric Kling–Gupta efficiency implemented as `kge_np`. Its components are mean-flow ratio beta, normalized flow-duration-curve agreement alpha_np, and Spearman rank correlation. The minimized loss is the distance of these three components from one, equivalent to one minus efficiency.

Preserve the meaning of the mean and flow-duration-curve components. Replace the exact predicted-rank calculation with a differentiable approximation so the temporal-rank component supplies a useful training signal. Sorting discharge values for the flow-duration curve can propagate gradients almost everywhere; exact ranking is the principal obstacle. Do not silently retain an ineffective hard-rank component or substitute ordinary Pearson correlation on discharge.

Calculate each basin's NPE components over its valid scored times, then average the valid basin losses with equal basin weight. Do not rank observations pooled across basins. Preserve existing MSE reduction behavior rather than silently changing its weighting to match NPE.

Exact NPE remains the evaluation measure. A soft approximation is a training surrogate, not a claim of an improved evaluation metric, uniquely identifiable HBV parameters, or guaranteed better held-out performance.

Use [Blondel et al. (2020), Fast Differentiable Sorting and Ranking](https://arxiv.org/abs/2002.08871) as a starting point. It develops regularized ranking operators with O(n log n) computation and O(n) memory. Section 6.3 trains using squared distance between target ranks and soft predicted ranks; it does not validate the complete proposed NPE loss. The simpler all-pairs sigmoid approximation is also discussed in the paper, but has quadratic cost. The hydrological reference is [Pool, Vis & Seibert (2018)](https://doi.org/10.1080/02626667.2018.1552002).

## Engineering discretion and experiments

The implementing agent should choose the ranking implementation, dependency strategy, softness/scaling convention, and exact treatment of observed ranks. These are not settled by the illustrative sigmoid formula discussed during discovery. Choose a practical, maintainable implementation for the project's PyTorch execution paths; do not assume theoretical complexity proves actual device performance.

Use bounded experiments when they help resolve these choices. Useful questions include whether the temporal-rank term produces meaningful gradients at representative discharge scales, how ties and zero flows behave, and whether softness makes ranks effectively hard or nearly constant. Small synthetic examples or representative existing workloads are sufficient when they answer the question; an exhaustive method tournament or large hydrological benchmark is not required.

Account explicitly for missing observations, too few valid observations, constant series, zero-flow denominators, finite negative discharge, and nonfinite predictions. Define when a basin cannot contribute and what happens if none can contribute; do not silently reward undefined scores or model failures. Avoid arbitrary numerical safeguards that change the objective without explanation. Finite-softness surrogates can differ from exact NPE even at a perfect discharge match, depending on the formulation; explain and test the chosen semantics rather than claiming exact parity.

Follow `implement-vision` for experimentation: when findings drive a meaningful decision, retain brief notes in `planning/experimentation/<vision-filename>.md`, linked to this vision. Record the question, observations and limitations, and resulting choice; discard prototypes without ongoing value. Experiments support, but do not replace, tests and independent review. Return to the human only if evidence requires changing the agreed outcome, not for routine reversible decisions.

## Evidence of completion

- MSE retains its existing masking and reduction behavior, and reconstruction pretraining remains intact.
- Soft NPE's components and basin aggregation have independently established numerical expectations, including the chosen policy for undefined inputs and tied flows.
- The rank term itself supplies a useful gradient in meaningful examples; successful backpropagation through only the other two components is insufficient.
- A focused end-to-end check shows soft NPE reaches the hybrid parameter head through HBV, with finite gradients and a real training update. Shared losses can also operate on discharge tensors without an HBV dependency.
- Loss selection and checkpoint round trips preserve the objective and its settings. Existing training/inference behavior remains covered.
- Exact evaluation metrics remain unchanged. Documentation explains how to select the loss, its dimensions and aggregation, softness/scaling, and its distinction from exact NPE.
- Use project-native `uv` execution, strongly typed Python, Ruff, ty, and focused meaningful tests. Report observed validation and material limitations; do not claim improved predictive skill without held-out evidence.
