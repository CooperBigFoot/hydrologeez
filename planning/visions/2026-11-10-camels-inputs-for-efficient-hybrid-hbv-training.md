# CAMELS inputs for efficient hybrid HBV training

Make the available CAMELS data easy and efficient to use for hybrid HBV training. Prepare broadly so experiments can choose their inputs without rebuilding source preparation for each feature combination. Explore and implement the best-supported physical layout, reader, and in-memory representation rather than assuming the traditional HBV layout must serve deep learning unchanged.

This vision implements [discussion #172](https://github.com/CooperBigFoot/hydrologeez/issues/172), under [Program #170](https://github.com/CooperBigFoot/hydrologeez/issues/170). Delivering and validating this work completes #172; publishing this vision does not. Scientific experiments and the broader Program remain separate.

## Make data available without choosing experiments

Make all available daily fields in the local CAMELS Daymet, Maurer, and NLDAS products available, together with static catchment attributes, discharge, and derived physical HBV inputs. The basin-mean source headers include precipitation, minimum and maximum temperature, solar radiation, vapour pressure, day length, and snow water equivalent. No separate thermal/longwave radiation field was found in these headers; do not invent one or silently introduce an external dataset. Investigate actual coverage, missing values, units, and product-specific semantics rather than treating matching headers as proof of interchangeable data.

Preserve product identity and source fields. Provide derived mean temperature and PET with explicit derivation and provenance. Existing preparation uses mean Tmin/Tmax and Oudin PET; preserve that baseline rather than silently replacing it with the paper's Hargreaves method. Adding scientific alternatives is not required to deliver this input foundation.

Experiments independently select:

- Meteorological features and static attributes supplied to the neural encoder. In particular, support supplying all three products' precipitation estimates simultaneously, without automatically averaging them.
- The physical P/T/PET inputs that drive HBV. These remain physical values in HBV units, separate from normalized neural features.
- Basin/date selections and authorized data pools for training, preprocessing, validation, and inference.

Broad availability does not mean loading every field for every experiment or prescribing a default scientific feature set. Retain access to existing static attributes without silently selecting the paper's subset. Preserve field metadata, basin identity, date alignment, missingness and observation flags; distinguish missing observations from synthetic reconstruction masks and padding. Do not silently impute invalid physical forcings. Keep genuinely missing targets out of losses and preserve caller control over warm-up and evaluation masks.

Preparation and reading must not silently fit normalization on held-out data. Inherit the Program's information boundary: alpha pretraining may use explicitly declared held-out-region meteorology, but held-out discharge must not enter training or selection. Attribute selection, including whether a field carries discharge-derived information, remains an explicit experiment responsibility rather than an implicit all-columns model input.

## Explore layout and access together

Start from the existing CAMELS sources, preparation, stores, and readers described in [CAMELS inputs](../../docs/camels.md), and the delivered [hybrid runtime](../../docs/hybrid-hbv.md). Reuse correct preparation logic and source data where suitable. A new physical representation is allowed when evidence supports it; a replacement or duplicated source archive is not presumed necessary. Keep existing traditional HBV use working.

The current daily store contains per-basin NumPy arrays accessed through memory maps. Its reader copies selected basin/date ranges and is intended to reuse selections. Documented repeated file-opening costs are relevant evidence, not proof that the layout is unsuitable: loading once into RAM could remove that cost from training. The current prepared P/T/PET representation is not the full raw meteorological feature collection.

Compare plausible reuse/extension and alternative approaches proportionately. Measure loading into RAM separately from repeated batch access, tensor conversion, and device transfer where applicable. Account for copies, preparation cost, and peak memory rather than reporting only a narrow disk-read time. Compare equivalent selections and outputs, and distinguish warm-cache results from cold-read claims. Avoid an exhaustive storage tournament or a new general data framework.

Implement a coherent layout and reader that efficiently supplies supervised discharge training, masked meteorological pretraining, frozen-encoder discharge training, and inference. Align configurable features and attributes with the runtime's ordered input contract. Support repeated access without reparsing raw text or unnecessarily reopening and copying the same data for each training step. Leave format, array organization, caching, batching mechanics, and public interface details to implementation investigation. Evidence-backed runtime changes are allowed; the synthetic interface is not immutable.

The intended training environment is a university VM requested with 15 CPUs, 60 GB RAM, NVIDIA L4 with 24 GB VRAM, and an attached 5 TB virtual disk. These are user-supplied target specifications, not verified hardware or disk-speed measurements. No remote connection, provisioning, or paid compute is required by this vision. Local measurements may establish the design while clearly recording their limits; obtain authorization before accessing or changing remote infrastructure.

For scale, 671 basins × 12,784 daily records × 10 float32 features occupy about 343 MB before masks, copies, discharge, or training state. This is an illustrative estimate, not the complete expanded dataset's measured footprint. Measure the actual chosen representation. Prefer simple resident-data access when it fits comfortably, keeping CPU dataset memory distinct from GPU activations and gradients. If the required workload exceeds available RAM, investigate and implement a suitable bounded-memory strategy rather than insisting on full residency. Do not build multiple loading paths speculatively.

## Delivery boundaries and evidence

Reusable preparation, layouts, readers, and necessary runtime integration belong in hydrologeez. Scientific experiment assembly, feature comparisons, masking-policy studies, training campaigns, and regional performance analysis belong in `uzh-results` and are not this work's goal. Small real-data integration runs and loading/batching measurements are validation, not a scientific campaign. Do not claim that additional inputs improve accuracy merely because they are available.

Observable success includes:

- All available daily source fields from the three products, static attributes, observations, and derived HBV inputs can be selected with explicit identities and consistent alignment. Verify preservation against source data, including missingness and product/date differences.
- A caller can select three precipitation inputs for the encoder while independently choosing valid physical HBV forcings; feature selection does not require rebuilding source preparation.
- Selected data can feed both hybrid training strategies and inference, with correct feature order, physical units, masking boundaries, and authorized preprocessing. Validate integration with bounded real-data examples rather than full experiments.
- Measured loading, repeated batch access, and peak memory support the chosen design. Report data sizes, selected fields, hardware, dtype, cache conditions, conversion costs, and remaining limits. Do not invent an unagreed throughput threshold or claim untested VM performance.
- Existing traditional HBV readers remain usable. Document preparation and practical reader use, the selected design's rationale, and any justified runtime changes.

Use project-native `uv`, strongly typed Python, Ruff, ty, and focused test-first development. Leave reversible engineering details to the implementing agent.

## Research context

Feng et al. (2023), [§2.3 and Table A1](https://doi.org/10.5194/hess-27-2357-2023), describes five meteorological inputs for its standalone streamflow LSTM and 35 static attributes. Its parameter-learning network uses historical forcings and attributes, but the PDF does not independently enumerate its exact meteorological channels. Physical HBV uses P/T/PET. Main experiments use Maurer, with additional Daymet experiments; this does not prescribe our input selection or require reproducing the paper.

Local evidence: `data/camels-us/raw/basin_timeseries_v1p2_metForcing_obsFlow.zip`, its basin-mean forcing readme and product headers, the existing prepared stores, and `/Users/nicolaslazaro/Desktop/delta_HBV_papers/02_Feng_2023_Ungauged_Regions_Climate_Change.pdf`. Source archives and local papers are not new repository dependencies to commit.
