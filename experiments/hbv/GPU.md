# Local Metal experiment

`gpu.py` uses MLX on the local Apple GPU, vectorizing over simulations and zones
while advancing days sequentially. Basin forcing is gathered for each candidate
inside each day; it is not expanded over all candidates on the host. Elementwise
expressions are compiled, but this is not a custom fused Metal time loop.

Run from the repository root:

```sh
NUMBA_NUM_THREADS=8 uv run --group experiments python -m experiments.hbv.gpu --basins 1000 --candidates 8 --days 365 --repeats 5
NUMBA_NUM_THREADS=8 uv run --group experiments python -m experiments.hbv.gpu --basins 1000 --candidates 4 --days 90 --repeats 5 --history
uv run --group experiments pytest tests/experiments/test_hbv_gpu.py
```

Metal does not support float64. Device inputs and calculation use float32; the
reference is the compiled float64 CPU. The errors include all requested states
and fluxes, not only outlet predictions. The scaled tolerance ratio divides
absolute error by `0.0002 mm + 0.00002 * abs(reference)`: a ratio at most one
satisfies the stated tolerance. This allows small accumulation errors on stores
while keeping sub-mm transfers tightly bounded. It is evidence for these inputs,
not proof that float32 never changes a threshold branch. Threshold equality and
available-water cases also have independent hand-calculated tests.

Input assumptions: valid resolved parameters, valid initial memory, zero-area
padding with finite admissible parameters, daily forcing in mm/°C/mm. No scientific
constants were changed. Candidate sharing extends parameterization as permitted by
the configuration vision; zone states remain independent. Returned histories are
initial memory followed by end-of-day memory, with same-day fluxes. All routing
schedule entries are retained for restart. Callers can batch simulations before
recording; the GPU helper itself does not write Parquet.

Timings include per-day evaluation and final `mx.synchronize()`. Device preparation
includes float32 conversion, allocation and synchronization. Apple unified memory
does not remove these costs. Host output copies are timed separately, and history
stacking is included in evaluation. First evaluations include compilation/cache
loading, not process/import startup. CPU serial and 8-thread parallel timings use
the same inputs and requested outputs. Metal peak is allocator memory, not process
RSS; incremental peak subtracts the prepared-input baseline. Prior GPU outputs are
released before repeat evaluation. Prepared inputs are reused for all repeats.

`evidence/gpu-prediction.json` and `gpu-history.json` are the aligned runs. Earlier
possibly contended measurements are excluded from repository evidence.
Neither synthetic agreement nor conservation establishes real-world skill or
verified compatibility with an HBV-light executable.

## Aligned results on Apple M4 Max (64 GiB unified memory)

Five repeats, no concurrent benchmark: prediction-only (1,000 basins × 8
candidates × 365 days) had median GPU 0.1259 s, CPU serial 0.3929 s, and
CPU 8-thread parallel 0.0521 s. Thus this GPU approach was 2.42× slower
than parallel CPU, despite beating serial CPU. History (1,000 × 4 × 90)
had GPU 0.0391 s, serial 0.0583 s, parallel 0.00918 s, plus 0.00654 s
to copy GPU outputs to host. GPU was 4.26× slower than parallel CPU here.

Prediction-only maximum outlet difference was 0.0000369 mm and final zone
store difference 0.000341 mm; the largest normalized tolerance ratio was
0.230. History maximum zone-store difference was 0.0000927 mm; all normalized
ratios were below 0.088. Incremental Metal peaks were 85.2 MB and 162.1 MB
respectively. These measurements do not select a universal winner: a custom
fused time kernel and larger candidate batches remain untested.
