# CAMELS static attribute preparation and runtime

Make all existing CAMELS-US static attributes available for general reuse through
prepared storage, a reader, and a labeled in-memory table. Clustering is one
possible consumer, not the organizing purpose of this interface.

This is a usable CAMELS-specific V0. Real experiments and workflow friction will
guide later revisions and any general interface; do not build a multi-dataset
framework or speculative extension points now. The deliverable is working data
preparation and reading, not an experiment report.

Implements [issue #137](https://github.com/CooperBigFoot/hydrologeez/issues/137).
Completing this vision completes that issue; publishing this document does not.

## Data and existing patterns

All seven source tables are present in `data/camels-us/raw/`: `camels_topo.txt`,
`camels_clim.txt`, `camels_soil.txt`, `camels_vege.txt`, `camels_geol.txt`,
`camels_hydro.txt`, and `camels_name.txt`. Each contains the same 671 basin IDs.
They contain numeric values, categorical values, and basin names. The raw folder
also contains the CAMELS attribute documentation in PDF and spreadsheet form.
Use available documentation for field meaning, units, and provenance rather
than guessing from names.

The existing [`hydrologeez.camels`](../../src/hydrologeez/camels.py) preparation
reads these tables to establish identities but does not retain the complete
attributes. Daily inputs use per-basin NumPy files and JSON metadata; the reader
materializes selections in caller basin order for reuse. See
[the CAMELS documentation](../../docs/camels.md) for the existing preparation and
selection workflow.

Follow the separation between physical storage, reading, and reusable runtime
values. Static attributes need not copy the daily inputs' per-basin file layout:
these are small heterogeneous tables rather than long homogeneous time series.

## Required behavior

- Prepare all seven tables from the existing local files without changing raw
  sources or acquiring new data.
- Provide a physical layout and reader alongside the prepared time-series
  workflow. Reading prepared attributes must not require parsing the raw tables
  again or loading daily records.
- Return a labeled table with basin IDs identifying rows and source attribute
  names identifying columns. Retain numeric and categorical values, including
  names, without forcing everything into a homogeneous numeric array.
- Preserve basin IDs as strings, including leading zeros; preserve source field
  names, categorical values, and the meaning of missing values. Retain available
  source and unit metadata so consumers can interpret the fields.
- Support selecting basins in the caller's order, aligned with time-series
  selections. A materialized table must be reusable without rereading storage.
- Document how to prepare and read attributes, including basin alignment.
  Documentation examples need not perform clustering or plotting.

Do not scale values, impute missing values, or encode categories into model
features. Those are consumer choices. Clustering, plots, new data acquisition,
and a general cross-dataset interface are outside this work.

## Evidence of success and implementation latitude

Verify that prepared attributes cover all source basin identities and fields.
Check representative numeric, categorical, and missing values against raw
sources, including string identity preservation. Demonstrate that an explicitly
reordered basin selection returns matching rows in the requested order and can
align with the existing time-series selection. Include meaningful focused tests
and validate the implementation using the repository's uv, Ruff, ty, and test
workflow.

The exact file format, table library, API spelling, and arrangement of prepared
files are implementation choices. Choose the simplest useful design for this
CAMELS-specific V0. The labeled-table runtime and faithful source semantics are
settled requirements; a general abstraction or simulation-specific array
runtime is not.
