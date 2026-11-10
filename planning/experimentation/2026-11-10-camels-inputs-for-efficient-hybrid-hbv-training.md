# CAMELS hybrid input layout

Implements the [published vision](../visions/2026-11-10-camels-inputs-for-efficient-hybrid-hbv-training.md).

## Question and decision

Can resident selections remove repeated training input overhead without replacing
the source archive or traditional reader? Compare per-basin and consolidated
product NumPy arrays, then measure the complete selected-data path.

Keep per-basin/product float32 NPY files, mmap only during selection, then reuse
resident tensors. Consolidated product arrays load faster in the bounded probe,
but repeated resident batches have identical cost. Per-basin arrays follow the
existing reader structure and support independently dated basin records directly.
The measured one-time cost does not justify a second consolidated loading path.
Both options support product-specific end dates; consolidation does not require
padding every product to Daymet's end date. This choice favors simple preparation
and lower observed probe RSS over fastest initial loading. No throughput threshold
was agreed or invented.

No general cache, dataset framework or speculative bounded-memory path is added.
Callers explicitly choose and reuse the authorized resident pool. Existing
observations/static stores remain authoritative and are not duplicated on disk.
No hybrid model/runtime behavior changed.

The user subsequently excluded obvious duplicates/all-zero payloads. Preparation
verifies zero SWE and exact Tmin/Tmax equality per basin, stores explicit constants
and aliases, and also aliases derived mean temperature when identical. It retains
all selectable source identities and missing-row masks. No remote data was fetched.
The first full nine-field prototype was discarded after validation; raw archives
were never modified.

## Measurements

Apple M4 Max, 16 logical CPUs, 64 GiB RAM, macOS 15.7.3 ARM64; NumPy 2.5.4,
PyTorch 2.15.0. Warm filesystem cache, other agents active; not cold disk or
university VM evidence. Target 15 CPU / 60 GB / L4 24 GB / 5 TB remains user-supplied,
not verified. CPU resident data is distinct from GPU activations/gradients.

[Independent layout probe](../../docs/camels-evidence/hybrid-layout-comparison.json):
128 real Daymet basins, 9 float32 fields, 58,908,672 bytes/layout. Both readers copy
the same six selected fields into basin/day/feature order. Five repeated loads:

| Selected days | Consolidated load | Per-basin load | Resident values |
| --- | --- | --- | --- |
| 3,652 | 3.0–9.4 ms | 11.2–58.0 ms | 11,218,944 bytes |
| 12,784 | 10.8–18.4 ms | 18.8–102.8 ms | 39,272,448 bytes |

500 resident copied 32×365×6 batches took about 3.4 ms with either layout.
Peak process RSS was 144/241 MB consolidated versus 130/189 MB per-basin. These
include imports, mapped pages and overlapping repeated allocations, not isolated
incremental dataset memory. Preparing both prototype layouts together took 2.39 s;
that is not an isolated preparation comparison. Probe excludes observations,
row-presence arrays and tensor transfer; it is not the complete production reader.

[Final reader measurement](../../docs/camels-evidence/hybrid-measurements.json):
full compact preparation took 33.52 s. Stored files total 677,117,109 bytes,
including metadata and NPY headers; NPY payload is 675,225,287 bytes including
source-row presence. Without deduplication, nine-field values alone would occupy
873,505,116 bytes at the actual product lengths. The complete selected store fits
comfortably within the specified RAM; this is not a training-state VRAM estimate.

128 basins, 1990–1999 inclusive, six named fields across all three products:
11,218,944 resident value bytes plus 5,609,472 mask bytes. Loading took 0.459 s.
Zero-copy `torch.from_numpy` took 0.082 ms; no dtype conversion is needed.
500 resident slice batches (32×365×6) took 0.605 ms; indexed copying took 9.19 ms.
One MPS batch transfer with synchronization took 19.13 ms, including first-use
cost. This is not CUDA/L4 transfer evidence.

The complete adapter with physical Daymet P/T/PET and observations loaded in
0.952 s and retained 52,355,072 bytes in arrays/tensors (excluding Python metadata
and date labels). The adapter keeps raw selected values/masks plus neural and
physical tensors, so it is explicitly not zero-copy end-to-end. Peak process RSS
was 494,043,136 bytes, including both selections, interpreter, PyTorch, mapped
pages, and bounded runtime tests. Batch slices share resident data; advanced
indexing and device moves allocate batch-sized copies. Repeated batches never
reparse source text or reopen store files.

## Behavior and limitations

The [full source scan](../../docs/camels-evidence/hybrid-source-audit.json) covers
all rows for all 671 attribute basins. Maurer ends in 2008; Daymet/NLDAS in 2014.
All SWE is zero; Maurer/NLDAS Tmin/Tmax are identical everywhere. Source Hr is 12,
contradicting readme zero. These are preserved facts, not inferred useful snow
observations or independent temperature extrema. No longwave field exists.

Synthetic tests independently check ordering, gaps, sentinels, constants/aliases,
product coverage, physical rejection, observation access and target masks.
Bounded native tests compare all seven source fields in all three products for
one basin/month, then exercise both LSTM/KDA encoders with direct discharge and
masked pretraining→frozen discharge plus inference on two basins/29 days.
The measured run has finite losses/gradients/predictions; no accuracy improvement,
full scientific experiment, VM throughput or long-sequence training claim follows.

Reproduce after preparation:

```sh
uv run --extra hybrid python -m scripts.camels_hybrid_measure \
  --meteorology data/camels-us/meteorology --observations data/camels-us/prepared \
  --evidence measurements.json
uv run --all-groups --extra hybrid pytest
```
