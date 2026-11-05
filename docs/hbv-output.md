# HBV output storage

`hydrologeez.hbv_light.output` provides `run_sweep`, `OutputStore` inspection and
`OutputStore.read`. It uses the existing `Forcing`, `Parameters` and `Memory`
objects. It does not choose scientific dates, eligible basins or parameters.

```python
from pathlib import Path
from hydrologeez.hbv_light.output import VARIABLES, OutputStore, run_sweep

# forcing, parameters and initial are existing HBV objects.
# Label every forcing basin and every Parameters row. Repeated parameter IDs
# identify the same library member evaluated in different basins.
store = run_sweep(
    Path("data/hbv-results"),
    forcing,
    parameters,
    initial,
    basin_ids=["01013500", "03241500"],
    parameter_ids=parameter_ids,
    start="1990-01-01",
    record_start="1991-01-01",
    variables=["discharge", "soil_moisture", "recharge"],
)
# Call run_sweep again with exactly the same arguments after interruption.
# Completed batches are skipped; the unfinished batch starts from initial memory.

store = OutputStore(Path("data/hbv-results"))  # no history arrays loaded
print(store.variables, store.basin_ids, store.parameter_ids)
print(store.start, store.stop, store.complete, store.completed_batches)
print(VARIABLES)  # names, units, dimensions, time semantics
selected = store.read(
    variables=["discharge"],
    basin_ids=["03241500", "01013500"],
    parameter_ids=["p42", "p7"],
    start="1995-01-01",
    stop="1996-01-01",
)
q = selected.values["discharge"]  # [date, simulation], requested order
print(selected.basin_ids, selected.parameter_ids, selected.dates)
# A timestamp comparison is the same read with a one-day [start, stop) range.
```

## Coordinates and variables

Dates are Gregorian integer ordinals, matching CAMELS, and ISO strings at the API
boundary. Ranges are start-inclusive, stop-exclusive. Warm-up days execute normally
but are not written. State values are **end of day**, not initial-state rows.
Daily fluxes use mm/day; storage uses mm. Zone quantities refer to zone area;
discharge, groundwater and routing quantities refer to basin area.

`VARIABLES` lists every snow/soil state, groundwater state, routing storage and
internal flux available from the model. Only selected variables are stored.
`Selection.metadata` carries units, dimensions and time semantics with values.

Selections use requested basin order, then requested parameter order. Each pair
is one simulation column. Unknown variables/IDs, unavailable pairs and dates
outside recorded coverage raise `ValueError`; no silent truncation occurs.
Zone and lag variables use packed columns: `zone_offsets` and `routing_offsets`
delimit each selected simulation. Zone index within a slice follows its forcing
zone order; routing index zero means arrival on the next day. Empty lag slices
are valid for MAXBAS=1. `Selection.parameters` retains resolved zone, groundwater
and MAXBAS values in selection order; its basin indices refer to requested basins.
Parameter IDs are caller labels, not a constraint that different basins must have
identical resolved zone parameters. Full parameter values are saved per batch.

## Execution and recovery

Defaults are **256 simulations × 365 days**, one sequential writer. Numba's
existing thread controls apply; this API does not create workers or change thread
counts. Shared forcing is never expanded over parameter sets. Callers own the
full parameter and initial-memory arrays; these and metadata grow with simulation
count. Histories do not: at most one batch/window result is retained. Discharge-only
uses `history=False`; other selections compute bounded full tile histories and
write only selected columns. No hydrological equations changed.

Each variable/batch is an uncompressed float64 NPY file. Window values append
in date order using NumPy format utilities and bounded streaming writes, not writer
mappings; each completed file is fsynced once. A single contiguous variable-tile
copy bounds writer buffering when model columns are strided. Reads mmap only relevant
files and copy requested slices, not full histories. Opening inspects JSON and
file sizes without loading arrays. Reading requires a complete run. Completion
status is a snapshot; reopen after a writer finishes.

A batch is written to a pending directory. Files and directory entries are fsynced
before an atomic rename publishes the batch. Resume discards only that store's
unfinished pending batch and recomputes it; committed batches stay unchanged.
Input fingerprints include forcing values/areas/offsets, resolved parameters and
initial memory. Identities, order, dates, variables and batch/window settings must
also match. Matching requires scanning inputs again, not trusting filenames.
Missing or truncated committed files cause an error rather than silently skipping
lost results. This detects incomplete writes, not arbitrary same-size corruption
or deliberate edits. Do not edit a store while reading or resuming it.

There is no overwrite mode: choose a separate destination to change a run.
A POSIX advisory lock rejects simultaneous writers and releases automatically on
process death. This implementation targets local POSIX filesystems with atomic
rename, flock and directory fsync; Windows and network/object-store semantics are
not supported. Fsync is not a guarantee against hardware failure or volatile device
caches (macOS F_FULLFSYNC is not used). After failure, inspect disk space and fix
the cause before resuming. A failure during initial metadata creation can leave
`metadata.tmp`; no simulation has completed then. Preserve/inspect that destination
and choose a new one rather than overwrite unknown contents.

## Capacity and measurement

See [production-path measurements](../experiments/hbv-output-storage/README.md)
and the [earlier comparison](../experiments/hbv-output-writing/README.md).
Bounded live result arrays are not a hard RSS limit: allocator retention, JIT,
shared inputs, file buffers and OS page cache also consume memory. More Numba
threads increase daily working memory. Multiple processes multiply these costs.
Estimate output bytes from recorded days × selected columns × 8, plus NPY headers,
parameters and metadata. Full multi-decade CAMELS histories can exceed a 5 TB disk.
No VM access or actual VM capacity/concurrency validation is claimed.
