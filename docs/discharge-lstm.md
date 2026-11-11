# Standalone discharge LSTM

Install `uv sync --locked --extra hybrid`. `hydrologeez.lstm` provides a single-layer causal LSTM mapping daily meteorology and caller-scaled static attributes directly to daily discharge `[day, basin]`, without a bucket model.

```python
import torch
from hydrologeez.hybrid.data import Normalization
from hydrologeez.lstm.model import DischargeLSTM, ModelConfig
from hydrologeez.lstm.runtime import Batch, Training

features = ("daymet.prcp", "daymet.tmin", "daymet.tmax")
values = torch.rand(2, 30, 3)
padding = torch.zeros(2, 30, dtype=torch.bool)
norm = Normalization.fit(values, padding, features)

config = ModelConfig(features=features, attributes=("elev_mean",), hidden=256, output_activation="softplus")
model = DischargeLSTM(config, norm)
batch = Batch(
    meteorology=values,
    attributes=torch.zeros(2, 1),
    padding=padding,
    discharge=torch.rand(30, 2),
    discharge_valid=torch.ones(30, 2, dtype=torch.bool),
)
training = Training(model, learning_rate=0.001, discharge_loss="mse")
training.train_step(batch)
prediction = training.predict(batch)  # [day=30, basin=2], strictly positive under softplus
training.save("discharge_lstm.pt")
restored = Training.load("discharge_lstm.pt")
```

## Inputs, output policy and checkpoints

- `DischargeLSTM` consumes normalized meteorological channels plus static attributes (`len(features) + len(attributes)` inputs per day), a single `nn.LSTM` layer of width `hidden`, and a linear scalar readout. Unlike `HybridHBV`, no missing/synthetic-hidden indicator channels are concatenated; active (non-padded) meteorological days must be finite.
- `output_activation` defaults to `"softplus"`, enforcing strictly positive smooth discharge without silent post-hoc clipping. Pass `"linear"` when signed raw readout values are explicitly intended.
- `Training` shares `hydrologeez.losses.masked_mse` and `soft_npe_loss` with `hybrid.runtime.Training`. Exclude warm-up days and unscored dates via `discharge_valid`.
- `hydrologeez.lstm.camels.batch_from_inputs` slices resident `CamelsInputs` into `lstm.runtime.Batch` while verifying feature and attribute order against `ModelConfig`.
- Checkpoints store configuration, feature normalization, model weights, objective settings and Adam state. Caller-owned static-attribute scaling, epoch counters and data-loader positions belong to the experiment.
