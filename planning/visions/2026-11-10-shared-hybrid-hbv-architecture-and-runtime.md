# Shared hybrid HBV architecture and runtime

Build reusable hybrid hydrological models in hydrologeez to explore whether learned meteorological representations improve parameter regionalisation. Include Kimi Delta Attention (KDA), which the user specifically wants to explore, and an LSTM baseline. Both must support direct discharge training and masked meteorological pretraining followed by frozen-encoder discharge training.

This vision implements the scope of [discussion #174](https://github.com/CooperBigFoot/hydrologeez/issues/174), under [Program #170](https://github.com/CooperBigFoot/hydrologeez/issues/170). Delivering and validating this work completes #174; publishing this document does not. It does not complete the broader Program or establish improved prediction in ungauged regions.

## Shared model, two training strategies

Provide three neural building blocks, differentiable HBV, and their training/inference runtime:

- **Encoder:** selectable KDA or LSTM, consuming meteorology and catchment attributes and producing learned representations. These are alternatives, not layers that must be stacked together.
- **Parameter head:** a feed-forward neural network mapping representations to valid HBV parameters.
- **Pretext head:** a separate feed-forward reconstruction head predicting masked meteorological values from encoder representations. It is used for pretraining, not discharge inference.
- **Differentiable HBV:** physical simulation driven by meteorological forcings and the predicted parameters. Preserve the existing HBV process structure, rather than replacing it with a discharge-predicting neural network.
- **Runtime:** reusable training, validation, inference, model-state persistence, and transitions between the training stages below. Implement the strategies, not merely model classes that leave all training to experiment scripts.

“Shared” means shared logic and implementation in hydrologeez. It does not mean jointly updated weights between experimental models.

**δHBV:** train encoder and parameter head end-to-end against observed discharge, propagating gradients through HBV. The reconstruction head is unused.

**αHBV (working name):** train encoder and reconstruction head with masked meteorological reconstruction; then freeze the encoder and train the parameter head against discharge through HBV. The reconstruction head is unused in this second stage. Freezing must preserve encoder parameters and inference behavior, not merely omit them from one optimizer. Gradients must still reach the parameter head through HBV.

This borrows the δHBV idea of learned parameterisation through a differentiable simulator. It is not a reproduction of Feng et al. (2023), Kimi K3, or a language model. No published component count, dynamic-parameter scheme, or claimed performance gain is adopted.

## Implementation direction and exploration

Use PyTorch primitives for the neural networks, automatic differentiation, and optimization. Start with PyTorch's LSTM and investigate reuse of FLA's KDA implementation before writing custom attention kernels. FLA (Flash Linear Attention) is the library; KDA is an architecture it implements. Do not substitute another FLA architecture for KDA merely because it is easier to integrate. Preserve required licenses and attribution when adapting source.

The implementing agent should explore concrete network sizes, representation and head wiring, masking schemes, losses, parameter mappings, state handling, and runtime organization. Start with small synthetic inputs and useful working examples, not a broad configurable model framework. Keep both encoders behind the small common interface actually needed by the heads and runtime. No particular layer count, hidden size, masking ratio, sequence length, or performance threshold is settled.

Static versus time-varying parameter outputs remain open to implementation exploration, not a requirement to support every variant. Prefer the simplest coherent baseline; do not silently change HBV process laws to accommodate dynamic parameters. Explain the selected information context: whether parameter estimates use preceding inputs or a complete supplied window. Keep pretraining and downstream representation use compatible, and avoid unacknowledged future-input access. These choices and their limits should be explicit in the delivered examples rather than inferred from an encoder name.

Core components belong in hydrologeez. Experimental assembly, large training campaigns, and regional performance studies belong in `uzh-results`. CAMELS storage integration and reader design are separate work; develop this runtime on synthetic inputs first and allow later data-layout evidence to motivate revisions. This scope does not require a new training framework, custom GPU kernels, language-model weights, global-attention layers, a fine-tuned-pretrained third strategy, or a broad architecture tournament.

## Physical and pretraining boundaries

The current HBV in `src/hydrologeez/hbv_light/` uses float64 NumPy/Numba arrays, fixed parameters, and explicit restart state. Wrapping it in PyTorch does not create an autograd graph. A differentiable tensor implementation is needed. Preserve established forward behavior and existing callers; use [the HBV description](../../docs/hbv-light.md) and current simulator as references. Its elevation–vegetation zones and nonlinear upper-groundwater response are not the parallel HBV components in the Feng paper.

Check numerical and gradient behavior deliberately. Hard thresholds, available-water limits, powers near empty stores, and parameter-dependent routing support need attention. Existing MAXBAS changes routing-memory length through `ceil(MAXBAS)`; treating it as a learned parameter requires a deliberate tensor representation. Forward parity alone does not demonstrate useful gradients. Surface material limitations rather than silently smoothing or replacing the physical model.

A practical pretext starting point is reconstruction of deliberately hidden observed meteorological values, with feature normalization and mean squared error only on hidden, valid, non-padding targets. Keep synthetic masking distinct from genuinely missing observations. Make hidden inputs distinguishable from legitimate values and prevent target leakage through input preparation. Masking patterns and loss refinements remain explorable; no published masking constants are requirements. Keep normalized neural inputs separate from the physical P/T/PET inputs supplied to HBV.

A forward recurrent encoder reconstructs from preceding context and visible same-day features; it does not automatically see future days. Reconstruction quality alone is not evidence that its frozen representations support useful HBV parameters. The frozen αHBV strategy is an experimental hypothesis, not a promised improvement.

For later regional evaluation, inherit #170's information boundary: unlabeled meteorology from held-out regions is permitted for αHBV pretraining and must be described as transductive access. Held-out discharge must not enter training or model selection. Keep the supervised δHBV comparator clean. Document the allowed meteorological pool and time coverage when assembling experiments; do not silently fit preprocessing on arbitrary evaluation data.

## Hardware and evidence of success

Prefer development and validation on the user's M4 Mac, using PyTorch CPU and MPS where supported. Avoid requiring an NVIDIA GPU to deliver this foundation. FLA's optimized KDA support on the Mac is unverified: investigate it, but a small plain-PyTorch KDA implementation adapted from the licensed reference is acceptable if it avoids an accelerator-specific dependency. Preserve KDA rather than substituting a different encoder, and do not build multiple execution backends without a demonstrated need.

The university NVIDIA L4 is currently inaccessible. The user can provide an NVIDIA GPU, including an RTX 4090 through Runpod with SSH access, if a concrete development limitation warrants it. This is a fallback, not required validation or authorization to rent resources. Explain the limitation and ask before requesting provisioning. Optimized CUDA kernels and CUDA benchmarking are optional; absence of CUDA testing alone does not block completion.

Validate the selected KDA path's forward and backward behavior locally where feasible. Record device, package versions, model/input sizes, numerical results, and relevant memory/performance limits. Distinguish small local correctness checks from demonstrated large-scale training performance; do not claim support for untested devices.

Observable success includes:

- Both LSTM and KDA run with the shared heads and HBV on synthetic inputs; direct training produces finite useful gradients and updates the intended trainable components.
- Reconstruction training scores only the intended targets and updates encoder/reconstruction weights. Its saved encoder can enter αHBV discharge training without changing frozen encoder behavior, while the parameter head learns through HBV.
- Differentiable HBV matches the established simulator on representative forward cases, retains mass-balance/state behavior, and has checked gradients away from nondifferentiable boundaries. Existing HBV functionality remains intact.
- Saved model and preprocessing state can reproduce inference with the correct encoder, feature order, parameter mapping, and physical inputs. Examples make stage transitions and state handling understandable without a CAMELS dependency.
- KDA forward/backward validation on the selected backend and remaining scientific or performance limits are reported. Neither NVIDIA access, improved regional accuracy, nor a particular speedup is required to complete this reusable foundation.

Use project-native `uv`, strongly typed Python, Ruff, ty, and focused test-first development. Leave reversible engineering details to the implementing agent; explain consequential departures from the selected baseline.

## Research starting points

- [Kimi Linear official repository](https://github.com/MoonshotAI/Kimi-Linear), which points to FLA.
- [FLA KDA layer](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/kda.py), [readable PyTorch recurrence](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py), and [backend installation guidance](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md). Inspected during discovery, not installed or benchmarked. Check and record an appropriate version rather than depending on moving source assumptions.
- [Zerveas et al. masked time-series reconstruction implementation](https://github.com/gzerveas/mvts_transformer): masking and masked-MSE reference, not evidence of frozen KDA/LSTM transfer to hydrology.
- [NeuralHydrology hybrid-model documentation](https://neuralhydrology.readthedocs.io/en/latest/usage/models.html#hybrid-model): useful reference for separating neural and physical inputs, not a required framework dependency.
- User-provided sources: `/Users/nicolaslazaro/Desktop/delta_HBV_papers` and `/Users/nicolaslazaro/Downloads/k3_tech_report (1).pdf`, especially K3 §2.1.1, Eq. (1). Local PDFs are context, not repository dependencies. KDA's learned recurrent memory is distinct from HBV water stores.
