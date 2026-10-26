# HBV-light: a worked mathematical description

## Model identity and limits

HBV stores water in snow, soil, and groundwater, then routes runoff to the
outlet. The daily model below uses the standard HBV-light structure [1, §2].
Snow and soil stores belong to elevation–vegetation zones; groundwater stores
and routing serve the whole basin.

The scope excludes lakes, glaciers, slope-aspect corrections, connected
subcatchments, and alternative groundwater structures, including the nonlinear
upper reservoir in [1, Eq. 8].

The daily rules retain the published process laws and specify timing and
allocation where the sources leave them open. These illustrative choices affect
discharge; agreement with an HBV-light executable remains unverified.

| Status | What this description uses |
| --- | --- |
| Published structure | Snow threshold, degree-day melt/refreezing, retention, nonlinear recharge, moisture-limited evaporation, two groundwater stores, three linear outflows, triangular routing [1, Eqs. 1–6]. |
| Historical evidence | Linear elevation adjustments, daily PET input, and support for noninteger routing duration [2, Part 1, pp. 3–7]. Current defaults remain unverified. |
| Illustrative daily choices | Rain at threshold equality; solid-snow retention basis; beginning-of-day soil moisture for recharge; soil overflow sent to groundwater; evaporation after recharge; percolation before runoff; proportional limiting of competing upper outflows; integrated fractional routing bins; explicit initial memory. |

## Time, inputs, parameters, and memory

Day $n$ is the interval $[t_n,t_{n+1})$, with $\Delta t=1$ day. A subscript $n$
on a store means its value at the **start** of that day; $n+1$ means its value
after all daily operations. Superscripts such as $a$ and $b$ label intermediate
stores. All water transfers are daily amounts in mm. Multiplication by
$\Delta t$ converts a parameter rate to an amount. The rules assume a daily
timestep.

For one zone, precipitation $P_n\geq0$ is a daily amount in mm, temperature
$T_n$ is a daily mean in °C, and potential evaporation $E_n\geq0$ is a daily
amount in mm. All inputs must cover the same interval. Missing or invalid forcing
requires an explicit treatment before simulation.

All parameters and forcings are finite. The domains below define the mathematical
rules; calibration ranges and software search bounds require separate choices.

| Symbol | Meaning | Units and domain |
| --- | --- | --- |
| $TT$ | Rain/snow and melt/refreeze threshold | °C, real |
| $CFMAX$ | Degree-day melt factor | mm °C$^{-1}$ day$^{-1}$, nonnegative |
| $CFR$ | Refreezing factor relative to melt | dimensionless, $0\leq CFR\leq1$ |
| $CWH$ | Liquid retention per unit solid snow | dimensionless, $0\leq CWH\leq1$ |
| $SCF$ | Snowfall correction factor (SFCF in [2]) | dimensionless, positive |
| $FC$ | Maximum soil storage | mm, positive |
| $BETA$ | Soil recharge exponent | dimensionless, positive |
| $LP$ | Soil fraction above which evaporation is potential | dimensionless, $0\lt LP\leq1$ |
| $PERC$ | Maximum upper-to-lower percolation rate | mm day$^{-1}$, nonnegative |
| $UZL$ | Threshold for fast upper-store outflow | mm, nonnegative |
| $K_0,K_1,K_2$ | Recession coefficients | day$^{-1}$, nonnegative |
| $B=MAXBAS$ | Base duration of triangular routing | days, $B\geq1$ |

Available-water limits keep stores nonnegative, including for large recession
coefficients. When active, they reduce outflow below the published linear demand.
$FC$ is a conceptual storage capacity whose value may differ from measured field
capacity [2, Part 2, p. 6]. $LP$ is dimensionless as required by [1, Eq. 4];
the historical manual labels it inconsistently.

| Component | Complete memory at start of day | Inputs | Outputs |
| --- | --- | --- | --- |
| Snow | Solid water equivalent $S_n\geq0$, retained liquid $L_n\geq0$, both mm | $P_n,T_n$ | Liquid soil input $I_n$; updated $S,L$ |
| Soil | Soil water $M_n\in[0,FC]$, mm | $I_n,E_n$ | Recharge $R_n$, actual evaporation $A_n$; updated $M$ |
| Groundwater | Upper $U_n\geq0$, lower $V_n\geq0$, mm | $R_n$ | Generated runoff $G_n$; updated $U,V$ |
| Routing | Scheduled future outlet amounts $h_{n,0},\ldots,h_{n,N-2}\geq0$, mm | $G_n$ | Outlet amount $Q_n$; updated schedule |

For a single zone, all depths refer to the basin area. A restart requires this
memory, the parameters, and the forcing position. Routing length $N$ and its
schedule are defined below.

## One zone: follow the water through a day

### 1. Snow: partition, melt or refreeze, then release liquid

Precipitation becomes snow when $T_n\lt TT$. Otherwise it is rain, including equality
in this convention. Define corrected snowfall $P_n^s$ and rainfall $P_n^r$ in mm:

```math
P_n^s=\begin{cases}SCF\,P_n&T_n\lt TT,\\0&T_n\geq TT,\end{cases}
\qquad
P_n^r=\begin{cases}0&T_n\lt TT,\\P_n&T_n\geq TT.\end{cases}
```

The modeled precipitation is $P_n^*=P_n^s+P_n^r$. Snowfall correction adjusts
external forcing for measurement errors and snow evaporation [1, p. 3316].
Snow evaporation is represented solely through this adjustment.

First add precipitation: $S^a=S_n+P_n^s$ and $L^a=L_n+P_n^r$.
The melt amount $m_n$ and refreezing amount $f_n$ are:

```math
m_n=\min\{S^a,CFMAX\max(T_n-TT,0)\Delta t\},
\qquad
f_n=\min\{L^a,CFR\,CFMAX\max(TT-T_n,0)\Delta t\}.
```

These are the degree-day laws [1, Eqs. 1–2] with explicit available-water limits.
Only one can be positive. At $T_n=TT$ both are zero. Update the phases:

```math
S^b=S^a-m_n+f_n,\qquad L^b=L^a+m_n-f_n.
```

For this example, retention capacity is $CWH\,S^b$, on a **solid-water-equivalent**
basis. Release excess liquid to soil, then retain the remainder:

```math
I_n=\max(L^b-CWH\,S^b,0),\qquad
S_{n+1}=S^b,\qquad L_{n+1}=L^b-I_n.
```

The sources leave the retention basis ambiguous between solid snow and total
pack water; the latter would give a different capacity. With no solid snow,
all rain passes directly to soil through $I_n$.

The snow balance is
$S_{n+1}+L_{n+1}-S_n-L_n=P_n^*-I_n$.
Melt and refreezing cancel because each is a transfer between phases.

### 2. Soil: divide input, enforce capacity, then evaporate

The recharge fraction increases with relative soil wetness [1, Eq. 3]. Here it
uses start-of-day soil moisture. Let $r_n$ be
the initial recharge amount, $o_n$ overflow, and $M^a,M^b$ intermediate soil
storages, all in mm except the dimensionless fraction:

```math
r_n=I_n\left(\frac{M_n}{FC}\right)^{BETA},\qquad
M^a=M_n+I_n-r_n,
```
```math
o_n=\max(M^a-FC,0),\qquad R_n=r_n+o_n,\qquad M^b=M^a-o_n.
```

Overflow joins groundwater recharge. This allocation and update order supplement
the published recharge relation and maximum storage $FC$. Evaluating recharge
with progressively updated moisture would change the partition for large inputs;
both approaches can conserve water.

After input and recharge, moisture limits evaporation [1, Eq. 4]. Cap its daily
amount by available soil water:

```math
A_n=\min\left\{M^b,E_n\min\left(\frac{M^b}{LP\,FC},1\right)\right\},
\qquad M_{n+1}=M^b-A_n.
```

At $M^b=LP\,FC$, the demand equals potential evaporation. Actual evaporation can
still be smaller if the available soil water is insufficient. The soil balance
is $M_{n+1}-M_n=I_n-R_n-A_n$.

### 3. Groundwater: percolate, then generate runoff

Recharge enters the upper store. Percolation $D_n$ is an internal transfer in mm.
Give it priority in this illustrative daily allocation:

```math
U^a=U_n+R_n,\qquad D_n=\min(PERC\,\Delta t,U^a),
\qquad U^b=U^a-D_n,\qquad V^a=V_n+D_n.
```

The published response has two upper outflows and one lower outflow
[1, Eq. 5]. Define unconstrained upper demands $d_0,d_1$ in mm from the **same**
post-percolation store, then limit them together:

```math
d_0=K_0\max(U^b-UZL,0)\Delta t,\qquad d_1=K_1U^b\Delta t,
```
```math
\lambda=\begin{cases}1&d_0+d_1=0,\\ \min\{1,U^b/(d_0+d_1)\}&d_0+d_1\gt 0,\end{cases}
\qquad q_{0,n}=\lambda d_0,\quad q_{1,n}=\lambda d_1.
```

Proportional limiting preserves the ratio of the two demands when water is
insufficient. At $U^b=UZL$, the fast demand is zero. Lower outflow includes today's
percolation:

```math
q_{2,n}=\min(V^a,K_2V^a\Delta t),\qquad
U_{n+1}=U^b-q_{0,n}-q_{1,n},\qquad V_{n+1}=V^a-q_{2,n},
```
```math
G_n=q_{0,n}+q_{1,n}+q_{2,n}.
```

The groundwater balance is
$U_{n+1}+V_{n+1}-U_n-V_n=R_n-G_n$.
The sources specify the maximum percolation rate and linear response laws;
priority and allocation remain open. Reversing priority changes the hydrograph.
The bound $V\leq PERC/K_2$ depends on initial storage: a fuller lower store can
exceed it, and $K_2=0$ makes the expression undefined.

### 4. Routing: delay runoff without losing its water

The published routing is a triangular weighted sum of generated runoff
[1, Eq. 6]. It includes a same-day contribution. Let $s$ be lag in days and
$w_B(s)$ a density in day$^{-1}$:

```math
w_B(s)=\begin{cases}
4s/B^2&0\leq s\leq B/2,\\ 4(B-s)/B^2&B/2\lt s\leq B,\\ 0&\text{otherwise}.
\end{cases}
```

With $N=\lceil B/\Delta t\rceil$, integrate each daily lag bin to obtain
dimensionless weights:

```math
c_j=\int_{j\Delta t}^{(j+1)\Delta t}w_B(s)\,ds,
\quad j=0,\ldots,N-1,\qquad c_j\geq0,\quad\sum_jc_j=1,
```
```math
Q_n=\sum_{j=0}^{N-1}c_jG_{n-j}.
```

For integer duration this is the paper's daily-bin triangle. The 2005 manual
supports noninteger MAXBAS [2, Part 1, p. 3] but leaves its fractional-bin
algorithm unspecified. Here, the last partial bin is integrated with zero density
beyond $B$. Rounding $B$ or sampling and renormalizing the triangle may yield
different weights. The daily weights define discrete routing; exact continuous
convolution for uniform intraday runoff generation would require a separate
derivation.

Routing memory tracks water still in transit. For $N\gt 1$, $h_{n,j}$ is water from
earlier generated runoff scheduled for outlet day $n+j$, in basin mm. Set
$h_{n,N-1}=0$ as an unsaved boundary value:

```math
Q_n=h_{n,0}+c_0G_n,\qquad
h_{n+1,j}=h_{n,j+1}+c_{j+1}G_n,\quad 0\leq j\leq N-2.
```

Routing water is $W_n=\sum_{j=0}^{N-2}h_{n,j}$, so
$W_{n+1}-W_n=G_n-Q_n$. For $B=1$ day, $N=1$, the schedule is empty,
$W_n=0$, and $Q_n=G_n$. A stored runoff history can also determine the schedule,
but includes water that has already left the basin.

## Elevation and vegetation zones

Zones contribute in parallel to shared groundwater stores.
Let zone $z$ have area $a_z\gt 0$ in km², basin area $a=\sum_z a_z$, and fraction
$\alpha_z=a_z/a$, with $\sum_z\alpha_z=1$. Each elevation–vegetation unit has
its own $S_z,L_z,M_z$ and executes the snow and soil rules above. This is the
standard arrangement in [1, p. 3317]. The alternatives with only snow zoned or
with separate upper groundwater stores are excluded.

Local snow, soil, input, evaporation, and recharge depths are per **zone area**.
The shared groundwater and routing depths are per **whole basin area**. Define
the transfers once:

```math
\bar R_n=\sum_z\alpha_zR_{z,n},\qquad
\bar A_n=\sum_z\alpha_zA_{z,n},\qquad
\bar P_n^*=\sum_z\alpha_zP_{z,n}^*.
```

Use $\bar R_n$ in place of $R_n$ in the shared groundwater update, followed by
basin routing.
For example, fractions $0.25,0.75$ and recharge $8,2$ mm give basin recharge
$3.5$ mm. Adding the zone depths would give an incorrect $10$ mm. On a 100 km²
basin the recharge volume is 350,000 m³.

Use one snow/soil parameter set per vegetation class, shared across its elevation
bands. Different vegetation classes may share the same set. Groundwater and
routing use one basin-wide set. Historical options allow vegetation parameters
to differ or be tied [2, Part 1, p. 12]; the sharing map varies by configuration.
The map used in the 2024 study remains unverified.

### Forcing adjustments and daily PET

Already zone-representative inputs need no second elevation adjustment. If using
reference-station inputs, the historical manual supplies linear corrections
[2, Part 1, p. 5]. Let $h_z$, $h_P$, and $h_T$ be zone mean elevation and the
precipitation and temperature reference elevations, in m. Let $PCALT$ be a
precipitation gradient in percent per 100 m and $TCALT$ a cooling gradient in
°C per 100 m. Supply both gradient values:

```math
P_{z,n}=P_{\mathrm{ref},n}
\left[1+\frac{PCALT(h_z-h_P)}{10000}\right],\qquad
T_{z,n}=T_{\mathrm{ref},n}-\frac{TCALT(h_z-h_T)}{100}.
```

The precipitation factor must be nonnegative in every zone; an invalid correction
requires revising its inputs. Positive $TCALT$ means cooling with elevation.
Apply the linear precipitation adjustment before snow partition, then apply
$SCF$ only to snowfall. For example, 500 m above both
reference elevations, $PCALT=10$ and $TCALT=0.6$ multiply precipitation by 1.5
and reduce temperature by 3 °C. Zero gradients leave the reference inputs unchanged.

Daily PET enters directly as nonnegative zone amounts $E_{z,n}$, with no further
temperature correction. A single basin series may serve every zone under an
explicit uniform-PET assumption. Document the provenance of zone-specific series.

An alternative in [1, Eq. 7] starts from climatological evaporation and corrects
for temperature anomaly. With reference daily climatological amount $E_{\rm clim,n}$,
climatological temperature $T_{\rm clim,n}$ in °C, and coefficient $CET$ in
°C$^{-1}$, it has the bounded form

```math
E_n=E_{\rm clim,n}\min\{2,\max[0,1+CET(T_n-T_{\rm clim,n})]\}.
```

The historical manual accepts monthly means interpolated to daily values, a daily
climatology, or one PET value per simulation day [2, Part 1, pp. 6–7]. The
climatology route requires interpolation, calendar/leap-day, and zone-temperature
conventions. It is an alternative to the prepared daily PET used here.

## Initialization, restart, observations, and conservation

Specify initial $S_z,L_z\geq0$, with $L_z\leq CWH_zS_z$, soil
$0\leq M_z\leq FC_z$, shared $U,V\geq0$, and every routing schedule entry.
A dry start sets them all to zero; its status as an HBV-light default is
unverified. An empty schedule assumes all pre-start runoff has already arrived.
A documented prehistory can instead initialize the schedule.

Warm-up advances specified initial stores and routing memory through earlier
forcings. Retain all resulting memory for the assessment interval and exclude
warm-up scores. Assess sensitivity to initial states and warm-up length. The
2012 paper mentions one year as sufficient in most cases [1, p. 3316]; the 2024
application uses five years [3, §4.1]. Convergence depends on the parameters,
particularly when recession is zero.

For restart at day $n$, save all zone stores, both shared groundwater stores, and
the routing schedule, together with parameters, zone areas/sharing, and the next
forcing date. This complete state and identical future inputs preserve future
results. Omitting stores or clearing the routing schedule can change them.
Changing MAXBAS changes the model and the meaning of routing memory.

Generated runoff $G_n$ reaches the outlet through routing. The observable $Q_n$
is daily outlet depth after routing. For basin area $a$ in
km² and an 86,400-second model day, convert it to interval-mean discharge in
m³ s$^{-1}$ using

```math
\overline q_n=\frac{1000\,a}{86400}Q_n.
```

Align measured daily means to the same interval. A 1 mm outlet amount on 100 km²
is 100,000 m³, or approximately 1.157407 m³ s$^{-1}$ averaged over the day.

Total modeled water on the basin area basis is

```math
X_n=\sum_z\alpha_z(S_{z,n}+L_{z,n}+M_{z,n})+U_n+V_n+W_n.
```

Adding the component balances cancels snow release $I_z$, recharge $R_z$,
percolation $D$, and generated runoff $G$, each with its proper area factor:

```math
X_{n+1}-X_n=\bar P_n^*-\bar A_n-Q_n.
```

Relative to uncorrected input $\bar P_n=\sum_z\alpha_zP_{z,n}$, the same balance
has an extra external adjustment $\bar P_n^*-\bar P_n$. Elevation adjustments
also change the forcing basis relative to a reference station. Both are external
forcing adjustments. Over multiple days, sum the right side and compare with
the difference of initial and final $X$, including water still in routing.
Different daily orders can satisfy the same balance.

## Worked day and boundary checks

Use one zone with start-of-day stores $(S,L,M,U,V)=(10,0,50,10,20)$ mm,
an empty routing schedule, and forcing $(P,T,E)=(6,2,2)$ in mm, °C, and mm.
Choose $TT=0$, $CFMAX=2$, $CFR=0.05$, $CWH=0.1$, $SCF=1$,
$FC=100$, $BETA=1$, $LP=0.5$, $PERC=1$, $UZL=12$,
$(K_0,K_1,K_2)=(0.2,0.1,0.05)$ day$^{-1}$, and $B=2$ days.

| Operation | Calculation | Result (mm) |
| --- | --- | --- |
| Snow input | All 6 mm is rain; melt $2(2-0)=4$ | $S'=6$, liquid before release $10$ |
| Snow release | Retain $0.1\times6=0.6$ | $I=9.4$, $L'=0.6$ |
| Soil recharge | $9.4(50/100)=4.7$; no overflow | $R=4.7$, soil before evaporation $54.7$ |
| Evaporation | $2\min(54.7/50,1)=2$ | $A=2$, $M'=52.7$ |
| Percolation | $\min(1,10+4.7)=1$ | $U^b=13.7$, $V^a=21$ |
| Upper outflows | $0.2(13.7-12)=0.34$; $0.1(13.7)=1.37$ | $U'=11.99$ |
| Lower outflow | $0.05(21)=1.05$ | $V'=19.95$ |
| Routing | $G=0.34+1.37+1.05=2.76$; $c_0=c_1=0.5$ | $Q=1.38$, next day's schedule $h'=1.38$ |

Primes in this table mean end-of-day stores. Initial water is $X_n=90$ mm;
final water is $6+0.6+52.7+11.99+19.95+1.38=92.62$ mm.
Thus $\Delta X=2.62=6-2-1.38$ mm. Omitting routing would wrongly report a
1.38 mm loss. On a 100 km² basin, today's outlet mean is 1.597222 m³ s$^{-1}$.
If the next generated runoff were zero, its outlet would still receive 1.38 mm.
A restart must retain $h'=1.38$ to preserve that arrival. Zero generated runoff
is a conditional routing check; groundwater may continue releasing water on a
dry day.

Boundary cases:

- **Empty and dry:** zero stores, zero precipitation, and any nonnegative PET
  give zero evaporation and discharge. Demands cannot create negative stores.
- **No snow or zero retention:** with $S=0$ and warm rain, all rain reaches soil.
  With $CWH=0$, all unfrozen liquid is released. Melt never exceeds solid water;
  refreezing never exceeds liquid. At $T=TT$, precipitation is rain and both
  phase-change demands vanish in this convention.
- **Dry soil under heavy input:** $M=0$, $FC=100$, $I=150$, $E=0$ give
  $r=0$, overflow 50, recharge 50, and final soil 100 mm. At $M=FC$, all
  incoming water is recharge before evaporation. $I=0$ needs no division by $I$.
- **Evaporation limited by water:** $M^b=1$, $FC=100$, $LP=0.5$, $E=100$
  give demand 2 but actual evaporation 1 mm and final soil zero.
- **Competing groundwater demands:** with $U^b=1$, $UZL=0$, and
  $K_0=K_1=1$ day$^{-1}$, demands are 1 mm each; $\lambda=0.5$ and
  each releases 0.5 mm. If percolation already consumed the upper water,
  both upper outflows are zero. At $U^b=UZL$, only the fast demand vanishes.
- **Fractional routing:** $B=2.5$ days gives weights $(0.32,0.60,0.08)$.
  An isolated 10 mm generated pulse produces $(3.2,6.0,0.8)$ mm on the
  current and next two days. After its first day, the schedule is $(6.0,0.8)$
  and routing water is 6.8 mm. Restarting with that schedule preserves the tail.
- **One zone or identical zones:** $\alpha=1$ recovers the single-zone case.
  Splitting a zone into identical copies with fractions summing to its original
  fraction preserves basin transfers; changing zone forcing need not do so.

## Application context and remaining source limits

Clerc-Schwarzenbach et al. (2024) [3, §4.1] provides the main application context:
daily HBV-light simulations, 200 m elevation bands, merging bands smaller than
5% of catchment area with neighbors, and five years of warm-up. These are study
settings. Its forcing comparisons include Hargreaves PET. The exact software
release, full spatial option, and vegetation configuration remain unspecified.
The [research discussion](https://github.com/CooperBigFoot/hydrologeez/issues/93#issuecomment-6428560036)
reports that the linked archive contains PET calculations and catchment
location/area data, with HBV configuration files absent. Archive access could
not be independently repeated here. These gaps prevent an exact reproduction.

The public 2005 manual is historical and explicitly incomplete. Current UZH
software and bundled help require registration and an emailed ZIP password [4];
no restricted software was obtained. Daily storage evaluation, overflow placement,
limited-water allocation, retention basis,
fractional routing, and initial history require comparison against a named
release and configuration. Fitting observed runoff alone leaves these choices
unresolved. The separate rain2flow implementation in
[Abbas et al. (2026)](https://doi.org/10.5194/hess-30-3399-2026) provides insufficient
evidence for HBV-light equivalence; its precipitation multiplier is excluded here.

## Mathematical dependencies

Zone snow and soil updates are independent given their own memory and inputs.
Their area-weighted recharge feeds groundwater, followed by fixed-weight,
finite-memory linear routing. Nonlinear recharge, thresholds, and available-water
limits couple successive days and prevent representing the full model as one
fixed linear convolution or affine recurrence.

Implementations must preserve transfer timing, allocation rules, and complete
restart memory. Changing a daily rule or replacing triangular routing with
reservoirs changes the model, even if water is conserved.

## Sources

1. Seibert, J. and Vis, M. J. P. (2012). *Teaching hydrological modeling with a
   user-friendly catchment-runoff-model software package*. HESS 16, 3315–3325.
   [Article](https://doi.org/10.5194/hess-16-3315-2012), §2, pp. 3316–3317,
   Eqs. 1–7. Primary source for the selected standard structure.
2. Seibert, J. (2005). *HBV light version 2, User's Manual*, November 2005.
   [Public historical manual](https://www.geo.uzh.ch/dam/jcr:c8afa73c-ac90-478e-a8c7-929eed7b1b62/HBV_manual_2005.pdf).
   Part 1 and Part 2 have separate page numbering. Historical details; current
   defaults and software compatibility remain unverified.
3. Clerc-Schwarzenbach et al. (2024). [HESS 28, 4219–4239](https://doi.org/10.5194/hess-28-4219-2024),
   §4.1. Application context with incomplete configuration details.
   [Linked archive](https://doi.org/10.5281/zenodo.10784701).
4. UZH, [HBV-light model information](https://www.geo.uzh.ch/en/units/h2k/Services/HBV-Model.html)
   and [download information](https://www.geo.uzh.ch/en/units/h2k/Services/HBV-Model/HBV-Download.html).
   Public access conditions checked during this work.
