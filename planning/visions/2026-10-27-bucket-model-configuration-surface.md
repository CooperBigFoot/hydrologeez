# Bucket-model configuration surface

## Purpose and authority

This is the agreed direction for a configuration-driven model surface, established
in [discussion #97](https://github.com/CooperBigFoot/hydrologeez/issues/97).
It completes that discussion's current design scope: an aspirational model
configuration contract illustrated in TOML, not a frozen or exhaustive schema.
Publication leaves the issue open. It prepares
[#91](https://github.com/CooperBigFoot/hydrologeez/issues/91) for model implementation;
it is not an execution-design or runtime-optimization vision and does not complete
#91. The requested handoff is to implement models against this surface, with
implementation scope established separately. Publishing this document implements
no model, calibration algorithm, or configuration loader.

The working pattern is: an agent writes an experiment configuration, runs it,
and results are saved for analysis. A notebook or direct Python model-object API
is not required. Prioritize large-scale work: thousands of basins, many simulations
and independent calibrations per basin, producing parameter datasets for
regionalization research. One selected model can serve many basins; do not interpret
this as requiring one mutable Python object per simulation.

Use predefined models following [the mathematical framework](../../docs/mathematical-framework.md).
There is no model-building language. HBV is first, using
[the documented daily model](../../docs/hbv-light.md), not an unverified promise
of exact compatibility with an HBV-light executable. Model classes, libraries,
solvers, batching, scheduling, precision, and memory layout are not specified here.

## Model setup versus experiment

Model configuration expresses the selected implementation, a name for the configured
setup, spatial subdivision, warm-up, and parameter sharing and assignment. The same
setup can be referenced by several experiments. `type` selects an implemented model;
`name` identifies the user's configured setup, not necessarily a different set of
physical equations. This supersedes the earlier discussion sketch in which `name`
alone selected the model implementation.

Experiments connect model setups with data, basin selections, splits, simulation or
calibration runs, repetitions/seeds, saved outputs, and metrics. Several model setups
can use the same experiment data. Their full schema is deliberately not designed.
Separate files and referenced tables are welcome; no monolithic experiment file or
fixed number of configuration layers is required. Dataset layout, forcing interfaces,
parameter-file layout, and metric selection remain separate design work.

## Zones and parameter sharing

Use **zone** (also called an HRU) for a spatial unit with separate snow and soil
memory. The supported subdivisions are:

| Configuration | Zones |
| --- | --- |
| Neither elevation nor vegetation subdivision | One whole-basin zone |
| Elevation only | One zone per elevation band |
| Vegetation only | One zone per vegetation class |
| Both | Each occupied elevation-band × vegetation-class intersection |

Omitting a subdivision means it is not applied. Vegetation classes can be source
categories or named groups of source categories. Omitting `groups` preserves the
source classes. A vegetation dataset reference names data supplied elsewhere; a
vector dataset is a useful example, not a selected storage contract.

Each zone has its own snow and soil states, even when parameters are shared.
Zone recharge is area-weighted into shared basin groundwater and routing. Their
parameters and memory remain basin-wide. An elevation-band minimum area fraction
applies to bands, not automatically to every intersected vegetation zone.
Map processing, band boundaries, small-band merging algorithms, and data preparation
are left for implementation discussion; do not expand this vision into their design.

Snow parameters are `TT`, `CFMAX`, `CFR`, `CWH`, and `SCF`; soil parameters are
`FC`, `BETA`, and `LP`. Groundwater/routing parameters are `PERC`, `UZL`, `K_0`,
`K_1`, `K_2`, and `MAXBAS` (the description's `B`).

For each snow/soil parameter, `by` describes the independent values within a basin:

| `by` | One independent value for |
| --- | --- |
| `basin` | All zones together |
| `elevation_band` | Each band, shared across vegetation classes |
| `vegetation_class` | Each vegetation class, shared across bands |
| `zone` | Each zone separately |

Sharing is configurable per parameter. Shared calibrated entries are **one
optimization variable**, not separate variables with identical bounds. Sharing
values never merges state. Group-shared calibration across different basins is
an experiment-level choice, separate from within-basin `by`.

**Extension to the current model description:** the existing document assigns one
snow/soil parameter set per vegetation class, shared across elevation bands, with
optional sharing between classes. This vision permits finer, per-parameter sharing
and independent zone/elevation values. Preserve the documented process rules, but
reconcile the parameterization text when implementing this extension. Do not claim
all these options are verified features of a named HBV-light software release.

## Illustrative TOML

The structure below anchors implementation discussion. It is not an existing loader
interface. Names and serialization can be refined without losing the stated meanings.
The numbers are illustrative settings, not scientific defaults or recommended search
bounds. The 200-m bands, 5% threshold, and five-year warm-up correspond to the 2024
study settings cited in the HBV description.

```toml
[model]
type = "hbv"
name = "hbv-200m-bands"
warmup_years = 5

[model.elevation_bands]
width_m = 200
minimum_area_fraction = 0.05

[model.vegetation_classes]
dataset = "veg_cls"

[model.vegetation_classes.groups]
forest = [1, 5, 7]
open = [2]
other = [3, 4, 6]

[model.parameters.TT]
by = "basin"
fixed = 0.0

[model.parameters.CFMAX]
by = "vegetation_class"
calibrate = true
bounds = [1.0, 8.0]

[model.parameters.CFMAX.overrides.forest]
fixed = 3.0

[model.parameters.FC]
by = "zone"
calibrate = true
bounds = [50.0, 500.0]

[model.parameters.BETA]
by = "elevation_band"
calibrate = true
bounds = [1.0, 5.0]

[model.parameters.K_2]
by = "basin"
fixed = 0.02
```

This intentionally shows selected parameters, not a runnable complete HBV setup.
The full configuration or referenced parameter sources must resolve the remaining
parameters; omission does not authorize invented values or bounds. `bounds` is an
illustrative elaboration of the agreed calibration assignment, not a selected
optimizer interface. Forest `CFMAX` is fixed; open and other each have an independent
calibrated value. The forest override replaces calibration, rather than retaining
both active assignments.

Warm-up means simulate the specified years before assessment, retain the resulting
complete memory, and exclude that period from assessment. Candidate parameter values
apply through warm-up. Initial states and restart specification remain to be designed;
no dry-start default is selected here. Complete memory includes routing, not only
water stores. Reusing warm-up memory from different parameters is not implied.

## Parameter assignments and external tables

Think of the resolved meaning as a table indexed by basin, zone or sharing group,
parameter, and run/parameter-set identity where applicable. It records an assignment
and references shared values. This is a conceptual description, not a requirement to
allocate a dense tensor or introduce a particular internal representation.

Assignments distinguish:

- **Fixed:** use a supplied value without optimizing or resampling it.
- **Loaded:** obtain predefined values from an external source, regardless of whether
  they came from previous calibration, manual selection, or another method.
- **Sampled:** sample once for a run and hold the value fixed during that run. This
  differs from randomly initializing a parameter that will then be optimized.
- **Calibrated:** optimize against observations. A calibration normally evaluates
  many candidate parameter sets; one simulation is not one calibration.

CSV or Parquet references avoid embedding all basin/run values in TOML. Parameter
sets must map to basins, tentatively via basin IDs, and support several sets per
basin. Specify assignments to the resolved shared parameter, not contradictory
values for zones that share it. Exact columns, IDs, distribution syntax, and join
mechanics remain open.

For example, a tentative basin-fixed override source is:

```toml
[model.parameters.overrides]
file = "fixed-parameters-by-basin.csv"
```

A minimal illustration for basin-shared parameters is:

```csv
basin_id,TT,K_2
basin_a,0.0,
basin_b,,0.02
```

A supplied override fixes that value for the matching basin. A blank cell leaves
the underlying assignment unchanged. In the follow-up-calibration use case,
unspecified parameters remain calibrated. Vegetation-, elevation-, and zone-specific
values need corresponding identities; this simple table does not define their format.

Use an explicit documented precedence: general assignment, then a named-group
assignment, then an explicitly selected basin-specific override source. Loaded
parameter collections are distinct from fixed-override tables; do not silently
interpret every data file as the same operation. Reject unresolved conflicts rather
than silently choosing between equally specific entries. The exact final schema
must document these rules and complete worked examples before claiming support.

## Workloads the surface must express

1. **Independent calibration ensembles:** 1,000 basins, 10 independent HBV
   calibrations per basin, with distinct seeds and retained calibrated parameter
   results suitable for a regionalization dataset. These are 10 optimization runs,
   not merely 10 candidate simulations.
2. **Group-shared calibration:** group basins by a supplied rule or mapping; fit one
   shared parameter set against the gauged basins in each group; compare against
   per-basin calibration. Basin forcing and memory remain separate. Objective
   aggregation and correspondence between heterogeneous zone structures are not
   decided here.
3. **Predefined simulations:** run supplied parameter values or external collections
   mapped to multiple basins. Their origin need not be calibration. Support multiple
   parameter sets per basin without writing them all in TOML.
4. **Random ensembles:** repeat simulations with sampled parameters; combine those
   with fixed values. Sampled values stay constant within each run. Sampling
   distributions and seed settings need explicit documentation when implemented.
5. **Follow-up partial calibration:** analyze ensemble results externally, produce a
   table identifying fixed values per basin, and recalibrate the rest. Different
   basins can fix different subsets, including none or all. If all parameters are
   fixed, skip optimization; ordinary simulation remains possible. No automatic
   parameter-selection analysis is required. Fewer free parameters reduce search
   dimension but do not guarantee a particular speedup.
6. **Reusable setups and comparisons:** use the same model setup in several
   experiments and compare multiple configured model setups on the same data.

Every state and model quantity must be accessible when requested, including full
state histories at every timestep, fluxes, and complete restart memory. Streamflow-only
runs must also be possible; access to everything does not require retaining every
history on every run or keeping it all in RAM. Results must preserve the identities
needed to associate them with basins, model setups, and parameter sets. Output-file
formats and metric APIs are not selected here.

## Documentation and evidence of success

LLMs will usually author the TOML. Prefer explicit, documented semantics over clever
shorthand. A fresh agent must be able to reproduce configurations without this
conversation. Document supported fields, units, domains, sharing, assignments,
references, precedence, and validation errors. Physical parameter domains come from
the model; experiment search bounds are separate choices.

When implementing this surface, provide complete examples for the workloads above
and a way to inspect the resolved assignments. Demonstrate that fixed entries are not
optimized, shared entries remain tied, sampled values stay fixed within a run, and
basin-specific overrides affect only their intended targets. Missing information or
contradictory assignments must not be silently repaired. Preserve full histories and
restart information when requested. These are observable semantics, not mandated
classes, test architectures, or execution strategies.

This vision deliberately stops short of a complete schema, CLI, calibration engine,
benchmark methodology, data preparation design, or universal backend. Refine concrete
syntax during implementation while preserving these meanings and updating examples.
The next discussion for #91 should establish its implementation deliverable using
this surface, rather than treating its older execution-oriented issue text as the
scope of this configuration work.
