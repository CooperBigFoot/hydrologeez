# Mathematical framework for bucket models

Describe bucket-type hydrological models as **interconnected dynamical systems with explicit water accounting and time conventions**. Use the same description style across models, while allowing each implementation to exploit its own mathematical structure.

This is an agreed foundation built from established state-space and component-based modeling, not a new mathematical formalism or a software API. It records the conclusion of [discussion #88](https://github.com/CooperBigFoot/hydrologeez/issues/88), within [Program #87](https://github.com/CooperBigFoot/hydrologeez/issues/87). Publishing this document does not close that discussion or authorize a model implementation.

## Describe each component

Specify four things:

| Part | Question |
| --- | --- |
| Memory | What information from the past determines future behavior given future inputs? |
| Inputs | What water and other signals enter the component? |
| Evolution | Which equations or ordered rules change its memory? |
| Outputs | What water does it release, and what quantities does it report? |

Include parameters and their domains, units, initialization, and time conventions. Memory includes routing queues, input history, or persistent modes when needed—not only named water stores. Distinguish complete memory from the amount of water it represents: a mode flag is not water, and past rainfall may include water that has already left.

For continuous-time components, the usual description is

$$
\dot{x}=f(x,u;\theta), \qquad y=h(x,u;\theta).
$$

For an ordered discrete model, it is

$$
x_{n+1}=F(x_n,u_n;\theta), \qquad y_n=H(x_n,u_n;\theta).
$$

Here $x$ is memory, $u$ inputs, $\theta$ parameters, and $y$ outputs. Expand these functions into the actual equations, branches, and intermediate quantities. A generic symbol $F$ is not a model specification. Retain a routing convolution, history law, or implicit equation when that is the clearer description; do not force every component into a finite-dimensional ODE.

For example, a threshold reservoir has memory $S$, inflow $u$, and discharge $q$:

$$
q=k\max(S-C,0), \qquad \dot S=u-q.
$$

The threshold is part of the output law. It does not require a different framework. General nonlinear laws fit the same description, but their domains and solution properties still need checking.

## Preserve the model's meaning

Start from the chosen published model and exact variant. Record unresolved differences between publications and reference code rather than silently selecting new physics.

- **Time:** preserve ordered daily recipes as discrete maps and continuous equations as coupled evolution. Neither is automatically interchangeable with the other. State the timestep and, for continuous models driven by interval data, how inputs vary within each interval.
- **Order:** name old, intermediate, and updated states. Specify threshold equality handling and allocation priorities when processes compete for limited water.
- **Initialization:** include all stores, modes, routing queues, and required prehistory. State any warm-up procedure separately.
- **Observations:** distinguish instantaneous rates, interval amounts, and interval-mean rates. Define the quantity compared with measured discharge and any area or unit conversion.

For example, continuous discharge rate $q(t)$ gives interval amount

$$
Q_n=\int_{t_n}^{t_{n+1}}q(t)\,dt.
$$

This is not generally the endpoint rate multiplied by the interval duration.

## Connect components through explicit transfers

Define each shared water transfer once and use the same amount or rate in its donor and receiver balances. Use a common volume basis or account for contributing areas when connecting water depths.

Internal transfers cancel in the whole-model water balance. Include water still in routing memory and any external gains or losses. Conservation is necessary, but does not by itself establish correct timing, nonnegative storage, or equivalence to the intended model.

**A component boundary is not necessarily a solver boundary.** Continuously interacting components share evolving transfer rates. Passing only a daily total may discard information needed downstream, even if each isolated calculation is exact. Solve coupled equations together or use a coupling method with declared accuracy. For a discrete recipe, preserve its specified transfer timing. Mixed continuous/discrete connections need an explicit timing convention.

## Keep description separate from execution

The mathematical description establishes the behavior. It does not mandate component classes, a universal solver, a compiler, or one execution algorithm.

An implementation may fuse several components into one loop or solve, or evaluate them in separate passes when their dependencies permit it. Optimize the complete model and calibration workload, not each component in isolation.

Look for structure actually present in the equations:

- parameter-independent calculations that can be reused;
- linear or affine recurrences and exact finite-time updates;
- finite-memory routing evaluated by queues or filters;
- reflected accumulations, such as a suitable snow recurrence;
- sparse coupling and independent basin–parameter simulations;
- a simple compiled sequential loop when more elaborate methods do not help.

A nonlinear reservoir generally has no fixed input-independent convolution kernel. Arbitrary function composition does not guarantee an efficient parallel scan. Compact notation alone produces no speedup; benchmark plausible alternatives against a strong faithful baseline.

Distinguish three kinds of change:

1. **Equivalent reformulation:** preserves outputs and corresponding future memory for the same admissible inputs, parameters, and initialization. Floating-point operation order can still change rounding.
2. **Numerical approximation:** solves the same continuous equations with a declared method and accuracy, checked against an appropriate reference.
3. **Model change:** alters laws, thresholds, timing, routing, or allocation rules. Similar calibrated discharge is not proof of equivalence.

Threshold smoothing, replacing an arbitrary hydrograph by a reservoir cascade, or changing an ordered daily update requires an explicit model-change or approximation statement—not an optimization label.

## Apply this foundation to the first model

Write one mathematical description containing:

1. Model identity, sources, and selected variant.
2. Time conventions, inputs, outputs, parameters, units, and admissible domains.
3. Complete memory and initialization.
4. Expanded component equations, connections, and update order or continuous coupling.
5. Observation equations and whole-model water balance.
6. Established computational structure and candidate implementation choices, clearly separated from model requirements.

Include a small worked example. Verify meaningful boundaries, water accounting, output timing, and restart behavior with complete saved memory. For continuous equations, check numerical accuracy; for discrete recipes, check the specified mapping. Conservation and a few matching runs alone are not a universal equivalence proof.

Success means that another developer can implement the model without guessing its time semantics, and can choose a model-specific execution strategy without changing its declared behavior. No backend, precision, solver, first-model variant, or performance winner is selected by this document.

## Established precedents

- [SuperflexPy](https://gmd.copernicus.org/articles/14/7047/2021/) uses reservoir equations, lag functions, and algebraic connections. Its execution choices are not requirements of this foundation.
- [MARRMoT](https://gmd.copernicus.org/articles/12/2463/2019/) is a useful precedent for explicit model equations; its standardized formulations are not promises of original-code equivalence.
- [Santos et al. (2018)](https://gmd.copernicus.org/articles/11/1591/2018/) illustrates continuous-state reformulation of GR4 and why structural and numerical changes must be distinguished from exact rewriting.
