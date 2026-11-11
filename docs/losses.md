# Discharge losses

Install the `hybrid` extra for PyTorch. `hydrologeez.losses` has no dependency on HBV or its trainer; any discharge-predicting network can use these functions.

```python
from hydrologeez.losses import masked_mse, soft_npe_loss

# prediction, observed: floating tensors [time, basin], same device/dtype
# valid: boolean [time, basin], false for warmup and unscored periods
loss = soft_npe_loss(prediction, observed, valid, softness=0.1)
loss.backward()
```

`masked_mse(prediction, target, selected)` accepts any matching dimensions and averages squared error over **all selected elements**, not basin means. It does not filter nonfinite values. Hybrid discharge training retains its existing finite-target filtering; reconstruction uses this same MSE on hidden valid meteorological values. The old `hybrid.data.masked_mse` import remains available.

## Soft NPE

`soft_npe_loss` calculates each basin separately over its selected, nonmissing times, then averages contributing basin losses with **equal basin weight**. For each basin:

- `beta = mean(prediction) / mean(observed)`.
- `alpha_np = 1 - sum(abs(sort(prediction)/sum(prediction) - sort(observed)/sum(observed)))/2`.
- `rho` is correlation between exact observed midranks (ties share their average rank) and soft predicted ranks.
- Minimize `norm([beta - 1, alpha_np - 1, rho - 1])`, equivalent to one minus NPE with the rank approximation.

Soft rank for prediction `i` is `sum_j sigmoid((prediction[i] - prediction[j]) / temperature)`, where `temperature = softness * mean(observed)`. The constant self contribution does not affect correlation. `softness` must be finite and positive; the default **0.1** is a practical starting value, not a tuned hydrological optimum. Common positive rescaling of both flows leaves the loss unchanged. Small softness can saturate gradients; large softness approaches correlation on discharge rather than ranks. A perfect match with irregularly spaced flows can have a nonzero loss at finite softness. The approximation is not exact Spearman correlation, and its optimum need not be the exact observed series.

Mean and flow-duration components retain their exact meanings. Sorting values has gradients almost everywhere; tied values and absolute-value corners use PyTorch subgradients. No epsilon is added to change zero denominators. Norm has a zero subgradient at an exactly zero loss.

### Undefined inputs

- NaN observations are missing even when selected. Unselected values never enter the calculation.
- Selected infinite or negative observations are errors. Selected negative or nonfinite predictions are errors, even for an observation-ineligible basin. Finite negative discharge is **not clipped**.
- Basins with fewer than two observed times or constant observed flows cannot contribute. This includes all-zero observations and hence zero observed mean. If none contribute, raise `ValueError`.
- In an eligible basin, constant predicted flows (including all zeros) raise `ValueError`; they are model failures, not grounds to remove that basin from the average. Varying flows may contain zeros. Nonnegative, nonconstant predictions necessarily have positive mean.
- Numerically collapsed/nonfinite rank correlation or a nonfinite loss raises rather than receiving a favorable score. A model initialized to constant discharge needs a different initialization or MSE training before using this objective.

### Cost and scientific limits

This is a pure PyTorch all-pairs sigmoid approximation: **O(n²) time and autograd memory per basin**. Processing basins sequentially still retains each basin's backward graph until backward. A single 3650-day float32 pair matrix is about 53 MB; total training memory is larger. Use bounded training windows/batches, not full multi-decade records without a memory budget. Windowing changes the scored objective and belongs to the training caller, not the loss. CPU and MPS were exercised; CUDA was not available. See [bounded experiments](../planning/experimentation/2026-11-11-modular-discharge-losses-with-soft-npe.md).

[Blondel et al. (2020)](https://arxiv.org/abs/2002.08871) gives faster regularized ranking operators and discusses sigmoid pairwise approximation. Their rank-training example does not validate this full NPE objective. [Pool, Vis & Seibert (2018)](https://doi.org/10.1080/02626667.2018.1552002) is the hydrological reference. Exact `metrics.evaluate(...)["kge_np"]` remains the evaluation/calibration metric, unchanged. No improved held-out skill or parameter identifiability is claimed.
