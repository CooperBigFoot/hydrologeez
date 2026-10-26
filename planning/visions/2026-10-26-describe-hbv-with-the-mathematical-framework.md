# Describe HBV with the mathematical framework

## Outcome

Produce a human-readable mathematical Markdown description of the published standard HBV-light formulation. The purpose is to demonstrate that [the agreed framework](../../docs/mathematical-framework.md) can describe a concrete hydrological model—not to select the final model for the research programme or implement a simulator.

This is the research-and-documentation deliverable for [discussion #93](https://github.com/CooperBigFoot/hydrologeez/issues/93), within [Program #87](https://github.com/CooperBigFoot/hydrologeez/issues/87). Delivering the description completes the scoped work of #93; publishing this vision does not. Its mathematical account will inform later model implementation and data-layout work, without selecting their architecture.

## Chosen model and scope

Use the standard formulation described by [Seibert & Vis (2012), section 2](https://doi.org/10.5194/hess-16-3315-2012): temperature-threshold snow partition, degree-day melt and refreezing, snow liquid-water retention, nonlinear soil recharge and moisture-limited evaporation, upper and lower groundwater stores with capped percolation and three linear outflows (one thresholded), and triangular runoff routing.

Describe daily simulation of one basin with one outlet. Explain the single-zone case first, then elevation–vegetation units: snow and soil calculations have separate memory in each unit; area-weighted recharge feeds shared groundwater stores. Explain forcing adjustments, parameter sharing, and the area basis of transfers from the selected sources. A single zone has area fraction one. Zones are parallel contributing areas, not a cascade of upstream and downstream basins.

Exclude explicit lake and glacier routines, slope-aspect corrections, alternative groundwater structures, connected subcatchments, calibration implementation, runtime optimization, and model software implementation. No universal component API, backend, solver, storage format, or hardware choice is needed.

The selected published formulation is not a claim of exact compatibility with an HBV-light executable or exact reproduction of a study. The user expects to discuss Jan Seibert's preferred variant with him later; this does not block the worked example.

## Sources and known limits

Prioritize primary publications and documentation from Jan Seibert's group, including [the UZH HBV-light material](https://www.geo.uzh.ch/en/units/h2k/Services/HBV-Model.html). Cite sources near the choices they support. Distinguish published variants and unresolved differences rather than combining them silently.

[Clerc-Schwarzenbach et al. (2024), section 4.1](https://doi.org/10.5194/hess-28-4219-2024) supplies the main application context: daily HBV-light, 200 m elevation bands, merging bands smaller than 5% of basin area, and a five-year warm-up. These are study settings, not mandatory constants for this description. The inspected public materials do not identify the exact release, spatial option, or vegetation-zone configuration. Its [linked archive](https://doi.org/10.5281/zenodo.10784701) contains PET calculations and catchment location/area data, not HBV configuration files.

The 2012 publication explicitly permits different spatial arrangements; only its standard snow-and-soil-per-zone arrangement is selected here. The public 2005 manual provides historical evidence, not current software defaults or zone-count limits. Current UZH software and bundled help require registration and an emailed ZIP password; none was obtained during discovery. Do not make access to that software a prerequisite for describing the published model.

[Abbas et al. (2026)](https://doi.org/10.5194/hess-30-3399-2026) identifies a separate Python implementation, rain2flow, with a precipitation multiplier. It is not evidence of equivalence to HBV-light and does not add requirements to this example. Further provenance is recorded in [the research discussion](https://github.com/CooperBigFoot/hydrologeez/issues/93#issuecomment-6428560036).

## Write for hydrologists

Place the description in the repository documentation at a clear model-specific path. Use ASD-STE100-inspired technical English without claiming formal compliance. Explain processes, introduce symbols and units before equations, and use small worked examples where useful. This is a model explanation, not a literature survey or software specification.

Apply the existing framework rather than inventing a second formalism:

- Identify complete memory, inputs, parameters and domains, evolution, and outputs for each component. Expand generic state-transition symbols into equations and ordered rules.
- Specify daily time conventions, old/intermediate/updated states, threshold handling, and allocation when processes compete for limited water. Distinguish an interval amount from a rate.
- Define each transfer once on a stated area basis. Show how zone contributions combine and how internal transfers cancel in the whole-model water balance.
- Include routing memory, complete initialization, and the distinction between initial values and warm-up. Define the outlet observation and discharge-unit conversion.
- Explain the chosen potential-evaporation input treatment. Do not silently impose the paper's monthly-climatology correction on daily PET inputs or mix alternative forcing treatments.
- Include a small worked example demonstrating water accounting and timing. Address meaningful boundaries and restarting from complete memory through mathematical explanation or worked calculations; a production simulator or test suite is not required.
- Separate established computational structure from candidate execution ideas. Keep that discussion proportionate; performance investigation belongs to later work.

The publications give central equations but may not fully specify an executable daily recipe. Investigate update order, finite-water limits, routing weights for noninteger duration, forcing corrections, and initialization. Clearly distinguish source-supported behavior from any necessary illustrative convention. A convention must not silently change published physics or be presented as software-equivalent. If a material ambiguity cannot be resolved, explain it and its consequences explicitly rather than concealing it behind a generic equation.

## Rendering and prose revision

Revise `docs/hbv-light.md`, delivered in [PR #95](https://github.com/CooperBigFoot/hydrologeez/pull/95), to correct GitHub math rendering and tighten the prose. This follow-up completes #93 once delivered and verified.

Use fenced `math` blocks for display equations in the current `docs/` Markdown files. GitHub's Markdown API reproduces loss of LaTeX backslashes inside the existing `$$` blocks, including row separators in `cases`; fenced `math` blocks preserve them. Keep `cases` row separators off line endings to avoid an extra backslash in the rendered payload. Check the rendered math payloads against the source and inspect the GitHub display where available. Local lint and calculation checks alone do not establish rendering correctness.

Keep the HBV explanation precise, concise, and readable for hydrologists. Use short, direct prose. Remove throat clearing, promotional language, contrastive negation, and commentary about the document's purpose or adherence to the framework. Introduce processes, symbols, and units where readers need them. Preserve equations, values, worked examples, daily conventions, and material source uncertainties. Consolidate repeated qualifications into short statements at the relevant choice. Keep software-equivalence limits explicit once. The revision changes presentation only.

## Evidence of success

A hydrologist can follow one day's water movement and understand what persists into the next day. A developer can identify the component equations, zone connections, complete memory, time conventions, and observations without reconstructing this discussion. Any remaining source ambiguity is visible and precisely bounded, not an implicit implementation choice.

The document demonstrates the framework on one coherent model, with traceable sources and a worked example. It does not claim numerical agreement with unavailable reference software, reproduce the 2024 calibration, or require choosing Jan's eventual preferred variant. Review the final description against the framework and these scope boundaries before reporting delivery.
