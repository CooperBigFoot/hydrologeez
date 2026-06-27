# GR6J golden oracle fixtures

Self-describing GR6J fixtures generated from the retired Rust `pydrology` oracle
(`/Users/nicolaslazaro/Desktop/work/pydrology`, prebuilt `_core` extension). Each
fixture stores its own generating params, so downstream tests are driven by the
fixture and never re-type constants. Parity target across the JAX rewrite is
~1e-4 relative (rtol=1e-4 + small atol), not bit-exact (see docs/DESIGN.md).

## Regeneration (deterministic, no auto-escalation)

Run from the pydrology environment, NOT the hydrologeez venv, NOT maturin:

    uv run --project /Users/nicolaslazaro/Desktop/work/pydrology \
        python <hydrologeez-worktree>/scripts/generate_gr6j_fixtures.py

The script reloads `gauge_id=camels_06224000/data.parquet`, builds
`ForcingData(time=df['date'], precip=df['mswep_precipitation'],
pet=df['potential_evaporation_sum_FAO_PENMAN_MONTEITH'])`, runs both param sets,
builds the Rust UH table, and emits the crafted step-branch artifact.

## Artifacts

### gr6j_camels_06224000.npz (canonical)
- Params (stored): x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5. warmup_length=365,
  basin_id=camels_06224000.
- The 20 flux arrays from `result.fluxes.to_dict()` (length 12333, no warm-up
  trim, no duplicate forcing keys): pet, precip, production_store, net_rainfall,
  storage_infiltration, actual_et, percolation, effective_rainfall, q9, q1,
  routing_store, exchange, actual_exchange_routing, actual_exchange_direct,
  actual_exchange_total, qr, qrexp, exponential_store, qd, streamflow.
- Coverage: drives `exponential_store` into [-30.30, -3.47] (AR = exp/x6 <= 0,
  the NORMAL / negative softplus regime). Exchange is zero (x2=0).

### gr6j_camels_06224000_exchange.npz (exchange-active)
- Params (stored): x1=350, x2=1.0, x3=90, x4=1.7, x5=0.5, x6=5. Same 20-flux +
  metadata schema as the canonical.
- VERIFIED on camels_06224000 (2026-06-27): F (exchange) nonzero on 12332/12333
  steps, range [-0.500, 0.046] (both signs). The negative-R routing clamp fires
  on 9550 steps (actual_exchange_routing != exchange) AND 2783 steps are
  non-clamp -- so ONE fixture covers both exchange-limb branches.
- `exponential_store` range [-4664.41, -3.47]: this regime drives Exp to the
  large-NEGATIVE AR clamp (AR = Exp/x6 reaches ~ -933, clamped to -33). This is
  an unphysical-but-numerically-sound oracle regime; relative-tolerance parity is
  safe because both impls run the identical f64 recurrence.

### actual_exchange_total -- resolution (known Rust deviation)
- Milestone-1 MATCHES Rust exactly: actual_exchange_total =
  actual_exchange_routing + actual_exchange_direct, EXCLUDES F. Confirmed
  total == routing + direct to 0.0 on the exchange fixture.
- airGR includes F in the total; this difference is a NOTED known deviation,
  deferred to a follow-up. Parity here is to the Rust definition.

### Parameter bounds note
- x6 uses the CODE bounds [1, 50] (constants.rs) for parity, not the documented
  [0.01, 20]. All bounds (code): x1 [1,2500], x2 [-5,5], x3 [1,1000], x4 [0.5,10],
  x5 [-4,4], x6 [1,50].

### gr6j_uh_ordinates.npz (true Rust UH table)
- Built from the genuine Rust binding `gr6j.gr6j_compute_uh_ordinates(x4)`
  (crates/pydrology-python/src/gr6j.rs), NOT the pure-Python mirror.
- x4_grid = [0.5, 1.0, 1.7, 2.0, 3.5, 5.0, 7.0, 10.0] (integer + non-integer).
- uh1_ord (8, 20), uh2_ord (8, 40); zero-padded beyond active support; each row
  sums to ~1. S-curves: SS1(i,x4)=(i/x4)^2.5 for 0<i<x4 else {0, 1};
  SS2(i,x4)=0.5*(i/x4)^2.5 for 0<i<=x4, 1-0.5*(2-i/x4)^2.5 for x4<i<2x4, else
  {0, 1}; D=2.5; uh1_ord[i-1]=SS1(i)-SS1(i-1), uh2_ord[i-1]=SS2(i)-SS2(i-1).

### gr6j_step_branches.npz (crafted positive-AR branch + +33 clamp)
- Built from the Rust binding `gr6j.gr6j_step(state, params, precip, pet,
  uh1_ord, uh2_ord)` (gr6j.rs), which accepts an arbitrary 63-element state
  (index 2 = exponential_store). Params [350,0,90,1.7,0,5]; UH ords from x4=1.7;
  precip=5.0, pet=2.0; UH states zeroed and x2=0, so q9=0 and F=0 and the exp
  store sees exactly the crafted exp_store (AR = exp_store/x6 is exact).
- Crafted exp_store / AR points (x6=5):
  - exp_store=40 -> AR=8   (large-POSITIVE softplus branch, QRExp=Exp+x6/exp(AR))
  - exp_store=50 -> AR=10  (large-POSITIVE softplus branch)
  - exp_store=165 -> AR=33 (positive branch + the +33 AR clamp boundary)
  - exp_store=200 -> AR=40 (positive branch + the +33 AR clamp)
- Stored: input_states (k,63), params (6,), precip[k], pet[k], uh1_ord[20],
  uh2_ord[40], oracle outputs qrexp[k] and exponential_store[k], and markers
  ar_target[k], is_positive_branch[k] (ar>7), is_ar_clamp[k] (ar>=33).
- This is the dedicated artifact covering the large-POSITIVE softplus branch
  (AR>7) and the +33 AR clamp -- the model-math branches the two run fixtures
  cannot reach (both keep AR<=0). With this, all softplus branches + both AR
  clamps + both exchange branches ship with non-vacuous Rust-oracle evidence.

## Reference constants (constants.rs)
- B=0.9 (UH1 split), C=0.4 (exp-store fraction), D=2.5, NH=20 (UH1 len; UH2=40),
  PERC_CONSTANT=(9/4)^4=25.62890625, MAX_TANH_ARG=13.0, MAX_EXP_ARG=33.0,
  EXP_BRANCH_THRESHOLD=7.0, STATE_SIZE=63.
