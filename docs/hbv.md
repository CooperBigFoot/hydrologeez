# HBV-Light (single zone)

HBV-Light is a 14-parameter daily lumped snow / soil-moisture / groundwater
rainfall-runoff model. hydrologeez implements the **single-zone (lumped)** variant
as a differentiable state-space model. Multi-zone (elevation-band) HBV is out of
scope.

## Parameters (14, canonical order, code bounds)

| # | Param | Meaning | Bounds |
|---|-------|---------|--------|
| 0 | `tt` | rain/snow temperature threshold [°C] | `[-2.5, 2.5]` |
| 1 | `cfmax` | degree-day melt factor [mm/°C/d] | `[0.5, 10.0]` |
| 2 | `sfcf` | snowfall correction factor [–] | `[0.4, 1.4]` |
| 3 | `cwh` | snow water-holding capacity [–] | `[0.0, 0.2]` |
| 4 | `cfr` | refreezing coefficient [–] | `[0.0, 0.2]` |
| 5 | `fc` | field capacity [mm] | `[50.0, 700.0]` |
| 6 | `lp` | ET limit (fraction of FC) [–] | `[0.3, 1.0]` |
| 7 | `beta` | recharge shape coefficient [–] | `[1.0, 6.0]` |
| 8 | `k0` | surface/quick recession [1/d] | `[0.05, 0.99]` |
| 9 | `k1` | interflow recession [1/d] | `[0.01, 0.5]` |
| 10 | `k2` | baseflow recession [1/d] | `[0.001, 0.2]` |
| 11 | `perc` | max percolation [mm/d] | `[0.0, 6.0]` |
| 12 | `uzl` | upper-zone Q0 threshold [mm] | `[0.0, 100.0]` |
| 13 | `maxbas` | routing/UH base length [days] | `[1.0, 7.0]` |

Only `maxbas` is hard-validated (`maxbas ∈ [1, 7]`, tied to the fixed length-7
routing buffer). The other 13 bounds are **advisory** — used for calibration
only; the model runs them silently out of range.

## State and initialisation

Single-zone state has `[B]` stores and a `[B,7]` routing buffer. Its per-basin
flat layout has 12 values, `[SP, LW, SM, SUZ, SLZ, b0..b6]`; time-stacked state
and flux leaves are `[B,T]`:

- `SP` snow pack [mm], `LW` liquid water held in snow (a.k.a. WC) [mm],
  `SM` soil moisture [mm].
- `SUZ` upper groundwater zone [mm], `SLZ` lower groundwater zone [mm].
- `routing_buffer` — the length-7 triangular-UH convolution delay line.

Initial state: `SM = 0.5 * fc` (the only non-zero, parameter-dependent init);
`SP = LW = SUZ = SLZ = 0`; routing buffer zeroed.

## Transition (per step)

For the lumped, no-elevation model the forcing (`precip`, `temp`, `pet`) passes
through unchanged. Each step reads the start-of-step stores and updates them in
this order — **all branches within a phase read the same start-of-step store
(explicit / simultaneous update, not sequential depletion):**

1. **Snow.**
   - *Partition:* `temp > tt` (strict) → all rain `(precip, 0)`; else → all snow
     `(0, sfcf*precip)`. `sfcf` scales snowfall only.
   - *Melt:* if `temp > tt`, `melt = min(cfmax*(temp - tt), SP)`, else `0`.
   - *Refreeze:* if `temp < tt`, `refreeze = min(cfr*cfmax*(tt - temp), LW)`, else
     `0`. (At `temp == tt`: snowfall, zero melt, zero refreeze.)
   - *Update:* `SP' = SP + p_snow - melt + refreeze`;
     `WC' = LW + melt - refreeze`; `lw_max = cwh*SP'` (pre-floor `SP'`);
     `outflow = max(WC' - lw_max, 0)` (and `WC'` is capped at `lw_max`); floor
     `SP', WC'` at 0. Then `snow_input = p_rain + outflow`.
2. **Soil.** All read the same start-of-step `SM`.
   - *Recharge:* `recharge = snow_input * clip(SM/fc, 0, 1)^beta`; `0` if
     `fc <= 0` or `snow_input <= 0`.
   - *Actual ET:* `lp_threshold = lp*fc`; `ET = pet` if `SM >= lp_threshold` else
     `pet*SM/lp_threshold`; capped `ET = min(ET, max(SM, 0))`; `0` if `fc <= 0` or
     `lp <= 0`.
   - *Update:* `SM_raw = SM + (snow_input - recharge) - ET`;
     `overflow = max(SM_raw - fc, 0)` is routed to upper-zone recharge (not
     discarded); `SM' = clip(SM_raw, 0, fc)`. The reported `recharge` flux is the
     total soil->upper-zone flux `recharge + overflow`.
3. **Response.** `q0`, `q1`, `perc` all read the same start-of-step `SUZ`; `q2`
   reads start-of-step `SLZ`.
   - `q0 = k0 * max(SUZ - uzl, 0)`; `q1 = k1 * SUZ`.
   - `perc = min(perc_max, max(SUZ, 0))`.
   - `SUZ' = max(SUZ + recharge_total - q0 - q1 - perc, 0)` (recharge_total =
     base recharge + above-FC overflow).
   - `q2 = k2 * SLZ`; `SLZ' = max(SLZ + perc - q2, 0)`.
   - `qgw = q0 + q1 + q2` (percolation is an internal SUZ→SLZ transfer, not in
     `qgw`).
4. **Routing.** `streamflow = convolve(routing_buffer, maxbas_weights, qgw)` (see
   below).

## MAXBAS routing — the masked-kernel crux

The routing is a triangular unit hydrograph of base length `maxbas`. A **fixed
length-7 masked kernel** (`ROUTING_BUFFER_SIZE = 7`, the declared `maxbas` upper
bound) preserves the routing semantics and batched tensor shape:

- Bin `i` integrates the triangle density over `[i, min(i+1, maxbas)]`. Bins with
  `i >= maxbas` collapse to zero — that is the mask. Tensor selection and clamp
  operations let Torch autograd differentiate the kernel with respect to
  `maxbas`.
- **Normalize-by-sum is load-bearing:** the raw per-bin weights integrate to 0.5,
  not 1.0, so the kernel is explicitly divided by its sum (`w / sum(w)` when
  `sum > 0`). Skipping this halves the routed flow.
- Ordinate **values** are continuous across integer `maxbas`, but the gradient has
  a **kink** at integer `maxbas` (a new bin activates exactly there, since the
  active-bin count is `ceil(maxbas)`). The kernel is value-continuous, not
  gradient-continuous — the same treatment GR6J gives integer `x4`.

The convolution is a **read-after-shift** length-7 delay line: the buffer is
shifted one slot, `weights * qgw` is injected, then the new head is read (the
same-day ordinate-1 term). There is **no forced one-step lag** — the routed pulse
turns on the same step `qgw` does — matching the published HBV-Light routing
(Seibert & Vis 2012; Seibert 2005 manual Eq. 6). It shares
`hydrologeez.convolution.convolve_delay_line` with GR6J.

## Fluxes (20 outputs, in order)

```
precip, temp, pet, precip_rain, precip_snow, snow_pack, snow_melt,
liquid_water_in_snow, snow_input, soil_moisture, recharge, actual_et,
upper_zone, lower_zone, q0, q1, q2, percolation, qgw, streamflow
```

`precip`/`temp`/`pet` echo the forcing; `snow_pack`, `liquid_water_in_snow`,
`soil_moisture`, `upper_zone`, `lower_zone` are post-update stores; the rest are
within-step fluxes; `qgw = q0 + q1 + q2`; `recharge` is the total soil->upper-zone
flux (base recharge + above-FC overflow); `streamflow` is the routed `qgw`
(same-day read-after-shift).

## Corrected version-fidelity note + retained deviations

The soil routine is **corrected** to the published HBV-Light: above-field-capacity
soil moisture is routed to upper-zone recharge (mass-conserving; Seibert & Vis
2012), not discarded, so the reported `recharge` flux is the total soil->upper-zone
flux (base recharge + FC overflow).

These behaviours are retained by decision and reproduced verbatim:

1. **Explicit-split over-draw.** `q0`, `q1`, `perc` all draw from the same
   start-of-step `SUZ`; the store is clamped at `max(0)` after subtracting all of
   them, but the already-emitted `q0`/`q1` are not reduced, so mass can be created
   on over-draw. The lower zone behaves the same for `q2`. Standard explicit-HBV
   (forward-Euler) behaviour, ported as-is.
2. **Only `maxbas` is range-validated.** The other 13 parameters run silently out
   of range.

## Usage

```python
import torch

from hydrologeez.models.hbv import HBVForcing, HBVModel

dtype = torch.float64
model = HBVModel(
    tt=torch.tensor(0.0, dtype=dtype),
    cfmax=torch.tensor(3.5, dtype=dtype),
    sfcf=torch.tensor(1.0, dtype=dtype),
    cwh=torch.tensor(0.1, dtype=dtype),
    cfr=torch.tensor(0.05, dtype=dtype),
    fc=torch.tensor(250.0, dtype=dtype),
    lp=torch.tensor(0.7, dtype=dtype),
    beta=torch.tensor(2.0, dtype=dtype),
    k0=torch.tensor(0.3, dtype=dtype),
    k1=torch.tensor(0.1, dtype=dtype),
    k2=torch.tensor(0.05, dtype=dtype),
    perc=torch.tensor(2.0, dtype=dtype),
    uzl=torch.tensor(20.0, dtype=dtype),
    maxbas=torch.tensor(3.0, dtype=dtype),
)
forcing = HBVForcing(
    precip=torch.tensor([[0.0, 4.0, 8.0, 2.0]], dtype=dtype),
    pet=torch.tensor([[1.0, 1.1, 0.9, 1.0]], dtype=dtype),
    temp=torch.tensor([[-2.0, 1.0, 3.0, 0.0]], dtype=dtype),
)

streamflow = model.run(forcing)
streamflow, fluxes, final_state = model.run(forcing, return_fluxes=True)
```

Use float64 on CPU for references and reproduction, and float32 on an explicitly
selected accelerator for training. Model execution preserves caller dtype and
device and does not mutate Torch global defaults.

## API reference

::: hydrologeez.models.hbv.model.HBVModel

::: hydrologeez.metrics
