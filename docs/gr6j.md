# GR6J

GR6J is a six-parameter daily lumped rainfall-runoff model. hydrologeez implements
it as a differentiable state-space model validated within ~1e-4 relative against the
retired Rust `pydrology` oracle.

## Parameters (code bounds, used for parity)

| Param | Meaning | Bounds |
|-------|---------|--------|
| `x1` | production store capacity [mm] | `[1, 2500]` |
| `x2` | groundwater exchange coefficient [mm/d] | `[-5, 5]` |
| `x3` | routing store capacity [mm] | `[1, 1000]` |
| `x4` | unit-hydrograph time constant [days] | `[0.5, 10]` |
| `x5` | exchange threshold | `[-4, 4]` |
| `x6` | exponential-store scale [mm] | `[1, 50]` |

`x6` uses the code bounds `[1, 50]`, not the documented `[0.01, 20]`; this is
chosen for oracle parity.

## State and initialisation

State = production store `S`, routing store `R`, exponential store `Exp` (may go
negative), and the two unit-hydrograph delay-line buffers `uh1[20]`, `uh2[40]`
(flat layout `[S, R, Exp, uh1, uh2]`, length 63). Initial state:
`S = 0.3*x1`, `R = 0.5*x3`, `Exp = 0`, buffers zeroed.

## Transition (per step), in oracle order

1. **Production.** If `P < E`: `tanh`-based evaporation `Es`. If `P >= E`:
   `Ps = x1*(1 - (S/x1)^2) * TWS / (1 + (S/x1)*TWS)`, with `Pn = P - E`,
   `Pr = Pn - Ps`, `Ws = min(Pn/x1, 13)` (tanh argument clipped at 13.0).
2. **Percolation.** `S = max(S, 0)`;
   `Perc = S * (1 - (1 + (S/x1)^4 / 25.62890625)^(-0.25))`,
   where `25.62890625 = (9/4)^4`. Total effective rainfall `= Pr + Perc`.
3. **UH split.** `uh1_input = 0.9*eff`, `uh2_input = 0.1*eff` (`B = 0.9`).
4. **UH convolution.** `q9` from UH1, `q1` from UH2 (delay line; the head is read
   AFTER the shift injects this step's input — the same-day ordinate-1 term,
   matching airGR MOD_GR6J; no forced one-step lag).
5. **Exchange.** `F = x2 * (R/x3 - x5)`.
6. **Routing store.** `routing_input = 0.6*q9` (`C = 0.4`); `R_tmp = R + 0.6*q9 + F`;
   clamp `R >= 0` (tracking `actual_exchange_routing`);
   `QR = R * (1 - (1 + (R/x3)^4)^(-0.25))`; `R -= QR`.
7. **Exponential store.** `exp_input = 0.4*q9`; `Exp += 0.4*q9 + F`;
   `AR = clamp(Exp/x6, -33, 33)`; three-branch softplus for `QRExp`
   (threshold 7; for `AR > 7`, `QRExp = Exp + x6/exp(AR)`); `Exp -= QRExp`.
8. **Direct branch.** `combined = q1 + F`; `QD = max(combined, 0)` logic.
9. **Total.** `Q = max(QR + QRExp + QD, 0)`.

## Unit hydrograph (the masked-kernel crux)

S-curves (`D = 2.5`):

- `SS1(i, x4) = 0` if `i <= 0`; `(i/x4)^2.5` if `i < x4`; `1` if `i >= x4`.
- `SS2(i, x4) = 0` if `i <= 0`; `0.5*(i/x4)^2.5` if `0 < i <= x4`;
  `1 - 0.5*(2 - i/x4)^2.5` if `x4 < i < 2*x4`; `1` if `i >= 2*x4`.

Ordinates: `uh1_ord[i-1] = SS1(i) - SS1(i-1)` for `i = 1..20`;
`uh2_ord[i-1] = SS2(i) - SS2(i-1)` for `i = 1..40`. Lengths are fixed at 20/40,
zero-padded beyond active support; `x4_max = 10`. The ordinates are a smooth
function of `x4`, so `jax.grad` w.r.t. `x4` is finite and continuous across integer
`x4`.

## Usage

```python
import os
os.environ["JAX_ENABLE_X64"] = "1"

import jax.numpy as jnp  # noqa: E402
from hydrologeez.models.gr6j import GR6J, GR6JForcing  # noqa: E402

model = GR6J(
    x1=jnp.asarray(350.0), x2=jnp.asarray(0.0), x3=jnp.asarray(90.0),
    x4=jnp.asarray(1.7), x5=jnp.asarray(0.0), x6=jnp.asarray(5.0),
)
forcing = GR6JForcing(precip=jnp.asarray(precip_series), pet=jnp.asarray(pet_series))
streamflow = model.run(forcing)
obs, fluxes, final = model.run(forcing, return_fluxes=True)
```

## Runnable example (executed at docs build, validated against the oracle fixture)

```python exec="true" source="material-block" title="gr6j_quickstart.py"
--8<-- "docs/examples/gr6j_quickstart.py"
```

## API reference

::: hydrologeez.models.gr6j.model.GR6J

::: hydrologeez.metrics
