# Zuko conditional flow model and training inputs

Build the reusable infrastructure needed to train conditional normalizing flows and sample parameter vectors from catchment attributes. The flow should feel like another model in hydrologeez, backed by Zuko rather than a custom flow implementation. Deliver the model adapter, training runtime, physical data layout, and reader so experiment scripts can use them directly.

This vision implements [discussion #171](https://github.com/CooperBigFoot/hydrologeez/issues/171), under [Program #170](https://github.com/CooperBigFoot/hydrologeez/issues/170). Implementing and validating this vision completes #171; publishing it does not. This is engineering infrastructure, not a scientific experiment or a claim about predictive performance or uncertainty.

## A small model backed by Zuko

Use Zuko's conditional distributions, density evaluation, and sampling. Wrap them to integrate our data, bounds, configuration, and checkpoints. Do not implement a custom flow, copy Zuko internals, or build an interchangeable-backend framework. Keep useful Zuko architecture settings configurable without exposing a new general plugin system; selecting the initial supported flow and precise settings is an implementation choice.

Parameter and attribute dimensions are constructor configuration, not hard-coded HBV constants. Each constructed model has fixed dimensions, while another instance can use different dimensions. A training example pairs a fixed-length parameter vector theta with a fixed-length numeric attribute vector c. The intended use is sampling HBV parameters conditioned on catchment attributes, but the flow must not encode HBV process structure, zone layout, or a fixed count of 13 parameters. Named, ordered columns provide the data contract; downstream code interprets the sampled vector.

Train by minimizing mean negative conditional log-likelihood using Zuko. Do not add Gaussian noise to training parameter vectors. Zuko's base-distribution noise belongs to sampling, not required training-data augmentation. Streamflow-gradient training and integration of HBV simulation into the flow training step are outside this work.

Accept explicit per-parameter lower and upper bounds and return samples in physical parameter units within those bounds. Training on bounded examples alone is not a guarantee of bounded output: provide the appropriate bounded distribution mapping, including density corrections when changing coordinates. Store bounds with the model. Do not silently adopt the hybrid head's ranges: calibration and hybrid HBV currently use different ranges. Resolve and document exact-boundary training-value handling, which can be problematic for invertible mappings to an open bounded interval. Do not silently alter stored examples or hide clipping inside the reader.

## Follow the existing data-access philosophy

Use the [hybrid input foundation](../../src/hydrologeez/hybrid/camels.py) and its [layout evidence](../experimentation/2026-11-10-camels-inputs-for-efficient-hybrid-hbv-training.md) as the precedent: select data once, keep the selected pool resident in CPU memory, and reuse it for training batches. Device transfer concerns the requested batch, not an implicit requirement to place the whole dataset on the accelerator. Do not introduce speculative streaming, multiple loading paths, caches, or a general dataset framework.

Design and implement a simple physical layout for parameter examples, their catchment association, and the metadata required to interpret them. Preserve parameter units, names/order, and catchment identities. Reuse existing prepared static attributes rather than creating another authoritative CAMELS attribute archive or duplicating attributes for every stored parameter example. Permit multiple examples per catchment without requiring equal counts. Attribute selection must remain explicit, with ordered numeric model inputs and no silent categorical encoding or missing-value imputation.

The reader should select parameter examples and attributes, align them by catchment identity, and build reusable resident inputs. Batches provide paired parameter and attribute tensors with clear example and feature axes. Preserve caller control over selected training and validation pools, batch order, and shuffling. Do not fit normalization implicitly during storage preparation, reading, or inference. Any fitted preprocessing must use an explicitly supplied pool, remain fixed during evaluation and sampling, and be carried with the checkpoint where needed to reproduce model inputs. Keep the existing convention that callers control static-attribute preprocessing; make that contract clear to experiment authors.

The exact file format, partitioning, in-memory representation, and API signatures are reversible engineering choices for the implementing agent. Investigate these together and validate that repeated batches do not reopen or reparse stored data. Prefer the simplest coherent path, informed by actual memory and access costs rather than an exhaustive storage comparison or an invented throughput threshold. The existing target environment is a user-specified university VM with 60 GB RAM and an NVIDIA L4 with 24 GB VRAM; it is not verified performance evidence or authorization to access remote infrastructure.

Training-example generation remains separate in [#173](https://github.com/CooperBigFoot/hydrologeez/issues/173). This infrastructure must accept supplied examples without selecting their generation recipe. The earlier issue text deferred layout work; this vision supersedes that deferral. Physical layout and readers are part of this delivery, not blocked on a scientific generation discussion.

## Familiar runtime and experiment ownership

Follow the ownership and naming conventions of the [hybrid runtime](../../src/hydrologeez/hybrid/runtime.py), without forcing shared inheritance or identical arguments:

- Construct a configured model and a small training object.
- Expose training steps and loss evaluation for supplied batches; callers own epochs, batch selection, and experiment orchestration.
- Provide an explicit conditional sampling method with documented sample, catchment, and parameter axes.
- Save and load model configuration, ordered fields, bounds, model-owned preprocessing, weights, and optimizer state so training can resume and inference can be reconstructed. Clearly document any caller-owned preprocessing and the limits of stochastic replay.

Reusable model, runtime, storage, and reader functionality belongs in hydrologeez. Parameter-example generation, scientific campaigns, and experiment assembly belong in uzh-results. A concise usage example should make the load, batch, train, evaluate, save/load, and sample path clear to agents writing those scripts. Keep existing hybrid and traditional HBV APIs working. Do not refactor other models into a common framework merely for surface consistency.

## Evidence of delivery

Use focused synthetic examples and bounded engineering checks, not a scientific experiment campaign, to show that:

- Different configured parameter and attribute dimensions work without HBV-specific model changes.
- Stored examples and selected attributes survive reading with correct values, ordering, identities, and unequal example counts per catchment.
- Resident batches feed finite conditional log-likelihood training and evaluation; sampling returns correctly shaped, finite, bounded physical parameter values.
- Bounds, coordinate transformations, and the documented endpoint policy behave correctly, rather than merely appearing bounded after clipping.
- Checkpoint loading preserves configuration and deterministic density evaluation, supports sampling, and restores optimizer state for resumed training. Do not require identical random draws without controlling random state.
- The documented end-to-end workflow runs through the reader and runtime. Report representative loading, resident-memory, and repeated-batching costs with their measurement limits; do not claim unmeasured VM throughput.

Use project-native uv execution, strongly typed Python, Ruff, ty, and focused test-first development. Leave routine engineering details to implementation. No scientific comparison with alpha/delta HBV, posterior interpretation, elevation-zone extension, or new training-example generation method is required.
