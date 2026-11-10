# Conditional parameter flows

Install `uv sync --extra flow`. `hydrologeez.flow` uses Zuko MAF conditional
normalizing flows; it does not simulate HBV or prescribe how examples are generated.
Each model fixes ordered parameter names, units, attribute names, and explicit
physical bounds. Hidden widths and transform count configure the MAF.

## Bounds and attributes

The model maps a physical parameter to `logit((theta - lower) / (upper - lower))`
and includes both affine and logistic Jacobians in its physical density.
Training and density evaluation reject exact endpoints, outside values, and
nonfinite values. The reader preserves supplied examples, including endpoints;
experiment authors must explicitly decide how endpoint observations should be
represented before training. No parameter noise, clipping, or automatic imputation
is applied. Sampling maps Zuko draws back through the logistic and affine maps.
The mathematical support is open; finite-precision sampling can round to a bound.

Static attributes are caller-preprocessed, ordered numeric finite columns from
`CamelsAttributes`. No normalization is fitted by storage, reading, training, or
sampling. If scaling is needed, fit it on an explicitly chosen training catchment
pool and apply the same fixed transformation to validation and sampling inputs.
Persist those caller-owned fitted values alongside the checkpoint; the checkpoint
does not reconstruct external preprocessing. The model owns only its fixed bound
transformation, stored with configuration and weights.

## Stored examples and resident batches

`save_examples` writes one uncompressed NPZ archive with `version`, `values`
[example, parameter], string `ids` [example], ordered `parameters`, and `units`.
Float32 or float64 values are retained. Row positions identify examples; repeated
catchments can have unequal counts. Attributes remain in the existing prepared
CAMELS store and are joined by string identity, including leading zeros.

`load_inputs` reads the archive once, selects catchments and ordered columns, and
keeps selected parameters, one attribute row per catchment, and example-to-catchment
indices in CPU memory. Selection currently decompresses no data but reads the full
values array before selecting rows; plan loading memory accordingly. Batches never
reopen files. Slices share parameter storage; index lists and attribute gathering
copy the requested batch. Only the batch moves to the requested device.

Callers choose disjoint catchment pools, row-level splits, epochs, and shuffle
order. Attributes follow the supplied catchment order; examples follow archive
order. Keep model and batch dtype/device consistent (`.double()` for float64
archives). Direct tensor calls must follow configured field order; resident
`batch` checks names and units against the model configuration.

## Example

Given supplied `examples.npz` and prepared CAMELS attributes (no generation recipe
is implied):

```python
from pathlib import Path

from hydrologeez.camels import read_attributes
from hydrologeez.flow.data import load_inputs
from hydrologeez.flow.model import ConditionalFlow, ModelConfig
from hydrologeez.flow.runtime import Training

ids = ("00000001", "00000002")  # explicitly selected training pool
attributes = read_attributes(Path("prepared/camels"), ids)
inputs = load_inputs(
    "examples.npz", ids, parameters=("capacity", "recession"),
    attributes=attributes, attribute_names=("area_gages2", "elev_mean"),
)
# Illustrative bounds in the archive's units, not prescribed HBV ranges.
config = ModelConfig(
    inputs.parameter_names, inputs.units, inputs.attribute_names,
    lower=(0.0, 0.0), upper=(1000.0, 1.0), hidden=(64, 64), transforms=3,
)
training = Training(ConditionalFlow(config))
batch = inputs.batch(config, examples=[0, 1])
training.train_step(batch)
loss = training.evaluate(batch)  # use your held-out pool for validation
training.save("flow.pt")
restored = Training.load("flow.pt")
samples = restored.model.sample(inputs.attributes, count=100)
# [sample=100, catchment=2, parameter=2], in physical units
```

Checkpoints contain ordered configuration, bounds, weights and Adam state.
They restore deterministic density evaluation and continued optimization, not RNG
state or identical random draws. They do not contain caller-owned attribute
preprocessing, selected pools, batch order, or epoch counters. Save those in the
experiment. Model construction and sampling use PyTorch randomness; control its
state explicitly if stochastic replay is required.

## Local access measurement

A synthetic warm-cache run on the development Apple Silicon Mac used 100,000
float32 examples, 13 parameters, 8 attributes and 1,000 catchments. The NPZ was
8,401,616 bytes; loading took 0.056 s. Resident tensors occupied 6,032,000 bytes.
The example identity tuple plus its strings added approximately 5,700,040 bytes
(`sys.getsizeof`, excluding other Python objects and allocator overhead).

For 1,000 repeated CPU batches of 256 examples, average construction time was
8.7 µs for a slice and 79.9 µs for an index list. This measures pairing/indexing,
not training, accelerator transfer, or end-to-end throughput. Imports and static
table preparation were excluded; full-process RSS and cold-cache costs were not
measured. Archive reading temporarily holds the full values array and selection
copies. These local numbers do not establish performance on the proposed 60 GB
RAM / NVIDIA L4 VM. Tests also remove the archive before repeated batching to
verify that no later reads occur.
