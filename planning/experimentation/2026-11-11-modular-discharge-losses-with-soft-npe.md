# Soft rank choice

Implements [the published vision](../visions/2026-11-11-modular-discharge-losses-with-soft-npe.md).

Question: is a dependency-free sigmoid approximation practical on the existing CPU/MPS paths, with useful rank gradients at discharge scales?

Choice: exact observed midranks and all-pairs sigmoid predicted ranks, temperature `0.1 * observed mean`. Unlike compiled CPU/CUDA ranking extensions or CPU/NumPy wrappers, this uses ordinary Torch operations on the existing MPS path. This trades the O(n log n) regularized rank algorithms discussed by Blondel et al. for a simple quadratic operator. No claim of a method tournament or superior performance.

Bounded local run (2026-11-11, PyTorch 2.15.0, float32, Apple M4 Max arm64, CPU 12 threads, Apple MPS): one basin of uniform synthetic flows in [0,10], a time permutation as prediction, forward + backward, one warmup then three timed repeats. MPS was synchronized immediately before and after each timed forward/backward. Mean seconds:

| Times | CPU | MPS |
| --- | --- | --- |
| 365 | 0.00071 | 0.00905 |
| 1095 | 0.00251 | 0.00412 |
| 3650 | 0.00597 | 0.00638 |

These are small local checks, not stable comparative benchmarks. CPU/MPS random draws differ. No CUDA check or peak-memory measurement. One 3650² float32 matrix alone occupies 53.3 MB; sigmoid intermediates and backward graphs add memory, accumulated across basins. Full-record/many-basin training can be prohibitive. Do not silently split or subsample inside the loss.

For observations [1,2,4] and predictions [1,4,2], beta and FDC agree exactly: only ranks contribute. At default softness, loss was 0.51014 and gradient norm multiplied by flow scale was 0.06255 on both CPU and MPS for scales 1e-4, 1, 1e4. This supports observed-mean scaling and a useful temporal gradient for this example, not all hydrographs. A separate independently worked sigmoid example and float64 gradcheck are retained as tests.

Ties use exact observed midranks; finite softness can bias the perfect-match score for irregular spacing. Very small softness can produce effectively hard ranks and large softness approaches Pearson correlation. Default 0.1 is not calibrated. Undefined observation series are omitted independently of model output; constant/zero predicted series are errors rather than omitted basins. No arbitrary denominator epsilon, improved evaluation metric, or held-out predictive benefit is asserted.
