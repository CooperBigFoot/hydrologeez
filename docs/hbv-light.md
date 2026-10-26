# HBV-light: a worked mathematical description

## Model identity and limits

HBV represents snow, soil moisture, groundwater, and delayed runoff with a small
number of stores. This description applies the [mathematical framework](mathematical-framework.md)
to the **standard structure in Seibert and Vis (2012), section 2** [1]: one basin,
one outlet, and a daily timestep. Start with one zone. The later zone section
repeats the snow and soil components over contributing areas and combines their
recharge into shared groundwater stores.

This is a hydrological explanation, not a simulator or a claim of agreement with
an HBV-light executable. It excludes lakes, glaciers, slope-aspect corrections,
connected subcatchments, and alternative groundwater structures. In particular,
the nonlinear upper-reservoir alternative in [1, Eq. 8] is not used.

The published equations do not specify every within-day operation. The ordered
rules below make **one illustrative daily convention** explicit. They retain the
published process laws but resolve allocation and timing where the inspected
sources do not. These choices can affect discharge. They are not evidence of
software equivalence, and later implementation work must not silently treat them
as reference-software behavior.

| Status | What this description uses |
| --- | --- |
| Published structure | Snow threshold, degree-day melt/refreezing, retention, nonlinear recharge, moisture-limited evaporation, two groundwater stores, three linear outflows, triangular routing [1, Eqs. 1–6]. |
| Historical evidence | Linear elevation adjustments, daily PET input, and support for noninteger routing duration [2, Part 1, pp. 3–7]. These are not claims about current defaults. |
| Illustrative daily choices | Rain at threshold equality; solid-snow retention basis; beginning-of-day soil moisture for recharge; soil overflow sent to groundwater; evaporation after recharge; percolation before runoff; proportional limiting of competing upper outflows; integrated fractional routing bins; explicit initial memory. |

## Time, inputs, parameters, and memory

Day $n$ is the interval $[t_n,t_{n+1})$, with $\Delta t=1$ day. A subscript $n$
on a store means its value at the **start** of that day; $n+1$ means its value
after all daily operations. Superscripts such as $a$ and $b$ label intermediate
stores, not additional memory. All water transfers below are **amounts over the
day**, in mm, not instantaneous rates. Multiplication by $\Delta t$ converts a
parameter rate to an amount; this does not authorize changing the timestep.

For one zone, precipitation $P_n\geq0$ is a daily amount in mm, temperature
$T_n$ is a daily mean in °C, and potential evaporation $E_n\geq0$ is a daily
amount in mm. They must describe the same interval. Missing or invalid forcing
is not zero rainfall. It needs an explicit data decision before simulation.

Use the following parameter domains for this explanation. They are mathematical
domains, not calibration recommendations or published software search bounds.
All parameters and forcings are finite.

| Symbol | Meaning | Units and domain |
| --- | --- | --- |
| $TT$ | Rain/snow and melt/refreeze threshold | °C, real |
| $CFMAX$ | Degree-day melt factor | mm °C$^{-1}$ day$^{-1}$, nonnegative |
| $CFR$ | Refreezing factor relative to melt | dimensionless, $0\leq CFR\leq1$ |
| $CWH$ | Liquid retention per unit solid snow | dimensionless, $0\leq CWH\leq1$ |
| $SCF$ | Snowfall correction factor (SFCF in [2]) | dimensionless, positive |
| $FC$ | Maximum soil storage in this recipe | mm, positive |
| $BETA$ | Soil recharge exponent | dimensionless, positive |
| $LP$ | Soil fraction above which evaporation is potential | dimensionless, $0<LP\leq1$ |
| $PERC$ | Maximum upper-to-lower percolation rate | mm day$^{-1}$, nonnegative |
| $UZL$ | Threshold for fast upper-store outflow | mm, nonnegative |
| $K_0,K_1,K_2$ | Recession coefficients | day$^{-1}$, nonnegative |
| $B=MAXBAS$ | Base duration of triangular routing | days, $B\geq1$ |

The finite-water limits below keep stores nonnegative even for large recession
coefficients. When a limit activates, the actual outflow is less than the
published unconstrained linear demand. Such parameter choices should not be
interpreted as evidence for the behavior of a particular HBV-light release.
$FC$ is a conceptual storage capacity, not necessarily measured field capacity
[2, Part 2, p. 6]. $LP$ is dimensionless as required by [1, Eq. 4]; some
historical manual prose labels it inconsistently.

| Component | Complete memory at start of day | Inputs | Outputs |
| --- | --- | --- | --- |
| Snow | Solid water equivalent $S_n\geq0$, retained liquid $L_n\geq0$, both mm | $P_n,T_n$ | Liquid soil input $I_n$; updated $S,L$ |
| Soil | Soil water $M_n\in[0,FC]$, mm | $I_n,E_n$ | Recharge $R_n$, actual evaporation $A_n$; updated $M$ |
| Groundwater | Upper $U_n\geq0$, lower $V_n\geq0$, mm | $R_n$ | Generated runoff $G_n$; updated $U,V$ |
| Routing | Scheduled future outlet amounts $h_{n,0},\ldots,h_{n,N-2}\geq0$, mm | $G_n$ | Outlet amount $Q_n$; updated schedule |

For a single zone, all depths refer to the basin area. No diagnostic flux needs
to be saved for a restart if this memory, the parameters, and the forcing
position are saved. Routing length $N$ and the meaning of its schedule are
defined below.

## One zone: follow the water through a day

### 1. Snow: partition, melt or refreeze, then release liquid

Precipitation becomes snow when $T_n<TT$. Otherwise it is rain, including equality
in this convention. Define corrected snowfall $P_n^s$ and rainfall $P_n^r$ in mm:

$$
P_n^s=\begin{cases}SCF\,P_n&T_n<TT,\\0&T_n\geq TT,\end{cases}
\qquad
P_n^r=\begin{cases}0&T_n<TT,\\P_n&T_n\geq TT.\end{cases}
$$

The modeled precipitation is $P_n^*=P_n^s+P_n^r$. Snowfall correction is an
external forcing adjustment, not a transfer from another store. The publication
uses it to account for measurement errors and snow evaporation, without an
explicit snow-evaporation flux [1, p. 3316]. Do not also subtract such a flux here.

First add precipitation: $S^a=S_n+P_n^s$ and $L^a=L_n+P_n^r$.
The melt amount $m_n$ and refreezing amount $f_n$ are:

$$
m_n=\min\{S^a,CFMAX\max(T_n-TT,0)\Delta t\},
\qquad
f_n=\min\{L^a,CFR\,CFMAX\max(TT-T_n,0)\Delta t\}.
$$

These are the degree-day laws [1, Eqs. 1–2] with explicit available-water limits.
Only one can be positive. At $T_n=TT$ both are zero. Update the phases:

$$
S^b=S^a-m_n+f_n,\qquad L^b=L^a+m_n-f_n.
$$

For this example, retention capacity is $CWH\,S^b$, on a **solid-water-equivalent**
basis. Release excess liquid to soil, then retain the remainder:

$$
I_n=\max(L^b-CWH\,S^b,0),\qquad
S_{n+1}=S^b,\qquad L_{n+1}=L^b-I_n.
$$

The sources describe retention as a fraction of snow water equivalent but do not
fully disambiguate solid snow versus total pack water. A total-pack basis would
give a different capacity; it is not silently substituted here. Rain passes
through immediately when there is no solid snow. There is no separate direct
rain-to-soil transfer to count again.

The snow balance is
$S_{n+1}+L_{n+1}-S_n-L_n=P_n^*-I_n$.
Melt and refreezing cancel because each is a transfer between phases.

### 2. Soil: divide input, enforce capacity, then evaporate

The published recharge fraction increases with relative soil wetness
[1, Eq. 3]. Evaluate it at **old** soil moisture in this recipe. Let $r_n$ be
the initial recharge amount, $o_n$ overflow, and $M^a,M^b$ intermediate soil
storages, all in mm except the dimensionless fraction:

$$
r_n=I_n\left(\frac{M_n}{FC}\right)^{BETA},\qquad
M^a=M_n+I_n-r_n,
$$
$$
o_n=\max(M^a-FC,0),\qquad R_n=r_n+o_n,\qquad M^b=M^a-o_n.
$$

Overflow is an explicit capacity convention: it joins groundwater recharge,
not a new direct outlet flow. The sources give the recharge relation and describe
$FC$ as maximum storage, but do not print this complete overflow/update recipe.
For large input, evaluating recharge using progressively updated moisture instead
would give a different partition even when neither recipe violates conservation.

After input and recharge, moisture limits evaporation [1, Eq. 4]. Cap its daily
amount by available soil water:

$$
A_n=\min\left\{M^b,E_n\min\left(\frac{M^b}{LP\,FC},1\right)\right\},
\qquad M_{n+1}=M^b-A_n.
$$

At $M^b=LP\,FC$, the demand equals potential evaporation. Actual evaporation can
still be smaller if the available soil water is insufficient. The soil balance
is $M_{n+1}-M_n=I_n-R_n-A_n$.

### 3. Groundwater: percolate, then generate runoff

Recharge enters the upper store. Percolation $D_n$ is an internal transfer in mm.
Give it priority in this illustrative daily allocation:

$$
U^a=U_n+R_n,\qquad D_n=\min(PERC\,\Delta t,U^a),
\qquad U^b=U^a-D_n,\qquad V^a=V_n+D_n.
$$

The published response has two upper outflows and one lower outflow
[1, Eq. 5]. Define unconstrained upper demands $d_0,d_1$ in mm from the **same**
post-percolation store, then limit them together:

$$
d_0=K_0\max(U^b-UZL,0)\Delta t,\qquad d_1=K_1U^b\Delta t,
$$
$$
\lambda=\begin{cases}1&d_0+d_1=0,\\
\min\{1,U^b/(d_0+d_1)\}&d_0+d_1>0,\end{cases}
\qquad q_{0,n}=\lambda d_0,\quad q_{1,n}=\lambda d_1.
$$

Proportional limiting preserves the ratio of the two demands when they compete
for insufficient water. It does not invent a priority between fast and slower
upper outflows. At $U^b=UZL$, the fast demand is zero. Lower outflow may use today's
percolation in this convention:

$$
q_{2,n}=\min(V^a,K_2V^a\Delta t),\qquad
U_{n+1}=U^b-q_{0,n}-q_{1,n},\qquad V_{n+1}=V^a-q_{2,n},
$$
$$
G_n=q_{0,n}+q_{1,n}+q_{2,n}.
$$

The groundwater balance is
$U_{n+1}+V_{n+1}-U_n-V_n=R_n-G_n$.
The sources specify a maximum percolation rate and the linear response laws, not
this priority or allocation rule. Reversing priority changes the hydrograph.
There is no unconditional bound $V\leq PERC/K_2$ for arbitrary initial memory;
for example, an initially fuller lower store can exceed it, and $K_2=0$ makes
that expression undefined.

### 4. Routing: delay runoff without losing its water

The published routing is a triangular weighted sum of generated runoff
[1, Eq. 6]. It includes a same-day contribution. Let $s$ be lag in days and
$w_B(s)$ a density in day$^{-1}$:

$$
w_B(s)=\begin{cases}
4s/B^2&0\leq s\leq B/2,\\
4(B-s)/B^2&B/2<s\leq B,\\
0&\text{otherwise}.
\end{cases}
$$

With $N=\lceil B/\Delta t\rceil$, integrate each daily lag bin to obtain
dimensionless weights:

$$
c_j=\int_{j\Delta t}^{(j+1)\Delta t}w_B(s)\,ds,
\quad j=0,\ldots,N-1,\qquad c_j\geq0,\quad\sum_jc_j=1,
$$
$$
Q_n=\sum_{j=0}^{N-1}c_jG_{n-j}.
$$

For integer duration this is the paper's daily-bin triangle. The 2005 manual says
noninteger MAXBAS is supported [2, Part 1, p. 3], but does not explain its fractional
bin algorithm. Integrating the last partial bin, with zero density beyond $B$,
is our explicit extension. Neither rounding $B$ nor sampling and renormalizing
the triangle is assumed equivalent. These daily weights define a discrete
mapping; they do not claim an exact continuous convolution for uniform intraday
runoff generation.

A schedule avoids confusing past runoff with water still in transit. For $N>1$,
$h_{n,j}$ is water from **earlier** generated runoff already scheduled for outlet
day $n+j$. Its units are basin mm. Define the sentinel $h_{n,N-1}=0$, not an
additional saved entry. Then

$$
Q_n=h_{n,0}+c_0G_n,\qquad
h_{n+1,j}=h_{n,j+1}+c_{j+1}G_n,\quad 0\leq j\leq N-2.
$$

Routing water is $W_n=\sum_{j=0}^{N-2}h_{n,j}$, so
$W_{n+1}-W_n=G_n-Q_n$. For $B=1$ day, $N=1$, the schedule is empty,
$W_n=0$, and $Q_n=G_n$. A stored runoff history can also determine the schedule,
but summing that history is **not** the routing water: some of it has already left.

## Elevation and vegetation zones

Zones are parallel contributing areas, not an upstream-to-downstream cascade.
Let zone $z$ have area $a_z>0$ in km², basin area $a=\sum_z a_z$, and fraction
$\alpha_z=a_z/a$, with $\sum_z\alpha_z=1$. Each elevation–vegetation unit has
its own $S_z,L_z,M_z$ and executes the snow and soil rules above. This is the
standard arrangement in [1, p. 3317], not the alternative with only snow zoned or
the alternative with separate upper groundwater stores.

Local snow, soil, input, evaporation, and recharge depths are per **zone area**.
The shared groundwater and routing depths are per **whole basin area**. Define
the transfers once:

$$
\bar R_n=\sum_z\alpha_zR_{z,n},\qquad
\bar A_n=\sum_z\alpha_zA_{z,n},\qquad
\bar P_n^*=\sum_z\alpha_zP_{z,n}^*.
$$

Use $\bar R_n$ instead of $R_n$ in the single groundwater update. Do not add raw
zone depths or average independently routed zone hydrographs. For example,
fractions $0.25,0.75$ and recharge $8,2$ mm give basin recharge $3.5$ mm, not
$10$ mm. On a 100 km² basin the recharge volume is 350,000 m³.

Separate memory does not imply separately calibrated parameters. For a concrete
sharing convention here, use one snow/soil parameter set per vegetation class,
shared across its elevation bands; permit different vegetation classes to share
the same set. Use one basin-wide groundwater and routing set. Historical
vegetation-parameter options support both differences and ties [2, Part 1, p. 12],
but do not establish a universal parameter-sharing map. This chosen map is not
claimed to reproduce the 2024 study or all HBV-light configurations.

### Forcing adjustments and daily PET

Already zone-representative inputs need no second elevation adjustment. If using
reference-station inputs, the historical manual supplies linear corrections
[2, Part 1, p. 5]. Let $h_z$, $h_P$, and $h_T$ be zone mean elevation and the
precipitation and temperature reference elevations, in m. Let $PCALT$ be a
precipitation gradient in percent per 100 m and $TCALT$ a cooling gradient in
°C per 100 m. Their values must be supplied, not inferred from zone count:

$$
P_{z,n}=P_{\mathrm{ref},n}
\left[1+\frac{PCALT(h_z-h_P)}{10000}\right],\qquad
T_{z,n}=T_{\mathrm{ref},n}-\frac{TCALT(h_z-h_T)}{100}.
$$

Require a nonnegative precipitation factor for every zone; do not silently clip
a physically invalid correction. Positive $TCALT$ means cooling with elevation.
This precipitation adjustment is linear, not compounded. Apply it before snow
partition, then apply $SCF$ only to snowfall. For example, 500 m above both
reference elevations, $PCALT=10$ and $TCALT=0.6$ multiply precipitation by 1.5
and reduce temperature by 3 °C. Zero gradients leave the reference inputs unchanged.

**This description consumes daily PET directly**, as nonnegative zone amounts
$E_{z,n}$. A single daily basin series may be assigned unchanged to every zone
if that assumption is stated. Different zone PET series require their own
provenance. No additional temperature correction is applied to these inputs.

An alternative in [1, Eq. 7] starts from climatological evaporation and corrects
for temperature anomaly. With reference daily climatological amount $E_{\rm clim,n}$,
climatological temperature $T_{\rm clim,n}$ in °C, and coefficient $CET$ in
°C$^{-1}$, it has the bounded form

$$
E_n=E_{\rm clim,n}\min\{2,\max[0,1+CET(T_n-T_{\rm clim,n})]\}.
$$

This is **not an extra step in the selected recipe**. The historical manual
accepts monthly means interpolated to daily values, a daily climatology, or one
PET value per simulation day [2, Part 1, pp. 6–7]. Using the climatology route
would also require explicit interpolation, calendar/leap-day, and zone-temperature
conventions. It must not be silently imposed on already prepared daily PET.

## Initialization, restart, observations, and conservation

Specify initial $S_z,L_z\geq0$, with $L_z\leq CWH_zS_z$, soil
$0\leq M_z\leq FC_z$, shared $U,V\geq0$, and every routing schedule entry.
A dry start sets them all to zero. It is one admissible choice, not a published
HBV-light default. An empty schedule means no pre-start runoff remains to arrive.
An alternative start can use a documented prehistory to fill that schedule.

Warm-up advances this same complete memory through earlier forcings before the
assessment interval. It is not an initial value and does not erase the need to
specify one. Keep routing memory at the end of warm-up; discard warm-up scores,
not its stores. Assess sensitivity to initial states and warm-up length rather
than assuming a universal duration. The 2012 paper mentions one year as sufficient
in most cases [1, p. 3316]; the 2024 application uses five years [3, §4.1].
Neither guarantees convergence for all parameters, especially zero recession.

For restart at day $n$, save all zone stores, both shared groundwater stores, and
the routing schedule, together with parameters, zone areas/sharing, and the next
forcing date. Reapplying the rules to this complete state and identical future
inputs gives identical future mathematics. Saving only groundwater, or clearing
the routing schedule, does not. Changing MAXBAS changes the meaning of routing
memory and is not an unchanged-model restart.

The observable $Q_n$ is an outlet **daily depth amount**, not $G_n$ and not an
endpoint discharge. For basin area $a$ in km² and a 86,400-second model day,
compare interval-mean discharge in m³ s$^{-1}$ using

$$
\overline q_n=\frac{1000\,a}{86400}Q_n.
$$

Align measured daily means to the same interval. A 1 mm outlet amount on 100 km²
is 100,000 m³, or approximately 1.157407 m³ s$^{-1}$ averaged over the day.

Total modeled water on the basin area basis is

$$
X_n=\sum_z\alpha_z(S_{z,n}+L_{z,n}+M_{z,n})+U_n+V_n+W_n.
$$

Adding the component balances cancels snow release $I_z$, recharge $R_z$,
percolation $D$, and generated runoff $G$, each with its proper area factor:

$$
X_{n+1}-X_n=\bar P_n^*-\bar A_n-Q_n.
$$

Relative to uncorrected input $\bar P_n=\sum_z\alpha_zP_{z,n}$, the same balance
has an extra external adjustment $\bar P_n^*-\bar P_n$. Elevation adjustments
also change the forcing basis relative to a reference station. Neither is an
internal source of conserved water. Over multiple days, sum the right side and
compare with the difference of initial and final $X$, including water still in
routing. Conservation alone cannot establish the correct daily order.

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
A restart retaining $h'=1.38$ preserves that arrival; one clearing it does not.
This conditional routing check does not assert that groundwater outflow actually
ceases on the next dry day.

Additional limits make the convention testable without a production simulator:

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
settings, not model constants. Its forcing comparisons include Hargreaves PET.
The inspected publication does not identify the exact software release, full
spatial option, or vegetation configuration. The linked archive was previously
inspected in the [research discussion](https://github.com/CooperBigFoot/hydrologeez/issues/93#issuecomment-6428560036)
and reported to contain PET calculations and catchment location/area data, not
HBV configuration files; archive access could not be independently repeated for
this description. No exact reproduction is claimed.

The public 2005 manual is historical and explicitly incomplete. Current UZH
software and bundled help require registration and an emailed ZIP password [4];
no restricted software was obtained. Source-supported equations alone do not
resolve daily storage evaluation, overflow placement, limited-water allocation,
retention basis, fractional routing, or initial history. Each resolution used
here is stated where it enters the calculation. A later reference comparison
must test these choices against a named release and configuration, not just fit
observed runoff. A separate implementation such as the rain2flow model discussed
in [Abbas et al. (2026)](https://doi.org/10.5194/hess-30-3399-2026) does not establish
HBV-light equivalence or authorize adding its precipitation multiplier here.

## Mathematical structure versus execution choices

Zone snow and soil updates are independent given their own memory and inputs;
they meet at the area-weighted recharge sum. Groundwater then feeds fixed-weight,
finite-memory linear routing. Nonlinear soil recharge and thresholded, limited
store updates still couple successive days. The full model is not one fixed
linear convolution or affine recurrence.

These are mathematical dependencies, not requirements for classes, a universal
component API, or separate software passes. A later implementation could fuse
operations, batch independent basins and parameter sets, or use a routing filter.
It must retain transfer timing, allocation rules, and complete restart memory.
Changing a daily rule or replacing routing by reservoirs is a model change,
not an optimization proved by conservation. Benchmarking and runtime selection
belong to later work; no backend, solver, or performance claim is selected here.

## Sources

1. Seibert, J. and Vis, M. J. P. (2012). *Teaching hydrological modeling with a
   user-friendly catchment-runoff-model software package*. HESS 16, 3315–3325.
   [Article](https://doi.org/10.5194/hess-16-3315-2012), §2, pp. 3316–3317,
   Eqs. 1–7. Primary source for the selected standard structure.
2. Seibert, J. (2005). *HBV light version 2, User's Manual*, November 2005.
   [Public historical manual](https://www.geo.uzh.ch/dam/jcr:c8afa73c-ac90-478e-a8c7-929eed7b1b62/HBV_manual_2005.pdf).
   Part 1 and Part 2 have separate page numbering. Historical detail, not current
   defaults or software-compatibility evidence.
3. Clerc-Schwarzenbach et al. (2024). [HESS 28, 4219–4239](https://doi.org/10.5194/hess-28-4219-2024),
   §4.1. Application context, not an exact configuration specification.
   [Linked archive](https://doi.org/10.5281/zenodo.10784701).
4. UZH, [HBV-light model information](https://www.geo.uzh.ch/en/units/h2k/Services/HBV-Model.html)
   and [download information](https://www.geo.uzh.ch/en/units/h2k/Services/HBV-Model/HBV-Download.html).
   Public access conditions checked during this work.
