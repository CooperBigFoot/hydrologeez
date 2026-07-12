# GR6J

GR6J is a six-parameter daily lumped rainfall-runoff model. hydrologeez implements
it as a differentiable state-space model.

## Parameters (code bounds)

| Param | Meaning | Bounds |
|-------|---------|--------|
| `x1` | production store capacity [mm] | `[1, 2500]` |
| `x2` | groundwater exchange coefficient [mm/d] | `[-5, 5]` |
| `x3` | routing store capacity [mm] | `[1, 1000]` |
| `x4` | unit-hydrograph time constant [days] | `[0.5, 10]` |
| `x5` | exchange threshold | `[-4, 4]` |
| `x6` | exponential-store scale [mm] | `[1, 50]` |

`x6` uses the code bounds `[1, 50]`, not the documented `[0.01, 20]`.

## State and initialisation

Each basin has production store `S`, routing store `R`, exponential store `Exp`
(which may go negative), and two unit-hydrograph delay lines. Stores are `[B]`,
and buffers are `uh1[B,20]` and `uh2[B,40]` (per-basin flat layout
`[S, R, Exp, uh1, uh2]`, length 63). Stacked trajectories add time on dimension
1. Initial state:
`S = 0.3*x1`, `R = 0.5*x3`, `Exp = 0`, buffers zeroed.

## Transition (per step)

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
zero-padded beyond active support; `x4_max = 10`. Their fixed sizes preserve the
delay-line semantics and batched tensor layout. The ordinates depend smoothly on
`x4`, so Torch autograd can differentiate them across integer `x4`.

## Usage

```python
import torch

from hydrologeez.models.gr6j import GR6J, GR6JForcing

dtype = torch.float64
model = GR6J(
    x1=torch.tensor(350.0, dtype=dtype),
    x2=torch.tensor(0.0, dtype=dtype),
    x3=torch.tensor(90.0, dtype=dtype),
    x4=torch.tensor(1.7, dtype=dtype),
    x5=torch.tensor(0.0, dtype=dtype),
    x6=torch.tensor(5.0, dtype=dtype),
)
forcing = GR6JForcing(
    precip=torch.tensor([[0.0, 5.0, 12.0, 8.0]], dtype=dtype),
    pet=torch.tensor([[1.0, 1.2, 1.1, 0.9]], dtype=dtype),
)

streamflow = model.run(forcing)
streamflow, fluxes, final_state = model.run(forcing, return_fluxes=True)
```

Constructors register their tensor arguments as `nn.Parameter` leaves, which
`run` uses by default. An explicit mapping must be complete; its values may be
scalar `[]`, per-basin `[B]`, or per-basin/per-time `[B, T]`:

```python
parameters = dict(model.named_parameters())
parameters["x2"] = torch.tensor([0.0, 0.5], dtype=dtype)
two_basin_forcing = GR6JForcing(
    precip=forcing.precip.expand(2, -1),
    pet=forcing.pet.expand(2, -1),
)
two_basin_streamflow = model.run(two_basin_forcing, parameters=parameters)
```

Warmup is separate forcing with the same batch size:

```python
warmup = GR6JForcing(
    precip=torch.zeros((1, 30), dtype=dtype),
    pet=torch.ones((1, 30), dtype=dtype),
)
streamflow = model.run(forcing, warmup=warmup)
```

Warmup runs without gradient recording and its final state is detached. The main
simulation remains differentiable.

## API reference

::: hydrologeez.models.gr6j.model.GR6J

::: hydrologeez.metrics
