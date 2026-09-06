# Conservative knowledge-flow skeleton: research assessment

Updated: 2026-09-05 UTC

Source: CONSERVATIVE_DYNAMICAL_KNOWLEDGE_FLOW_HIGH_LEVEL.md.
Scope: high-level assessment and analytic checks; no implementation or new
training experiment. The source document is unchanged.

## Assessment

The skeleton is worth developing because it connects exchange, a measurable
potential, local admissibility, and equilibrium. The strongest idea is that
useful individuality should survive through task/dynamics constraints, rather
than defining success as zero distance between every client's full system.

The resulting research question is:

> What task-relevant properties of heterogeneous graph dynamics can be
> exchanged, and which admissible exchanges reduce harmful propagation
> disagreement while preserving useful local behavior?

This is a research hypothesis, not a proved learning mechanism or an established
novelty claim. Functional Maps, generators, semantic probes, barycenters, and
pairwise flows all remain candidates. No explicit shared/private architecture
or residual decomposition is selected.

## What the skeleton already clarifies

- Feature differences affect initial conditions, while graph differences affect
  propagation. These influences interact along nonlinear trajectories; fitted
  generators are not pure measurements of structure.
- Representation time t and federated exchange time tau are different axes.
- Cross-client comparison requires comparable semantics and coordinates.
  A learned transport is needed only if the chosen knowledge object requires
  it; invariant or already shared observables may suffice.
- A constrained low-disagreement equilibrium can retain useful local
  differences. This is a better target than unrestricted consensus.
- A flux should be assessed by its realized effect on local systems, rather
  than merely by its size or by a decrease in an aligned-summary objective.

## Conservation: state the conserved object and the phase

Write x_m for a transported knowledge object during an exchange phase.
For fixed weights and coordinates,

$$
\dot x_m=J_m,\qquad \sum_m p_mJ_m=0
\quad\Longrightarrow\quad
\sum_m p_mx_m=\text{constant}.
$$

This preserves a weighted first moment. It does not preserve all information,
mutual information, task accuracy, diversity, or physical energy. For example,
two scalar summaries (+1,-1) and (0,0) have the same mean and different
dispersion. Ordinary averaging already has this invariant.

Conservation therefore needs a task-level reason. If the initial aggregate
encodes a harmful bias, exact preservation can also prevent improvement.
Depending on the knowledge object's geometry, it may be appropriate to
conserve a selected statistic K(x), rather than every entry of x. Linear
addition is not automatically meaningful for distributions or manifold-valued
objects.

Local learning generates new summaries. A whole-training balance would have
the form

$$
\dot x_m=S_m^{learn}+J_m,\qquad
\frac{d}{d\tau}\sum_m p_mx_m=\sum_m p_mS_m^{learn},
$$

when weights and coordinates are fixed. Changes in participation weights or
transport coordinates introduce additional terms. A defensible first claim is
conservation within a frozen exchange phase, not invariance over all training.

## Dissipation: harmful disagreement needs a task meaning

The illustrative mean-attraction law is mathematically consistent:

$$
\dot x_m=\kappa(\bar x-x_m),\qquad
E=\tfrac12\sum_m p_m\|x_m-\bar x\|^2
\quad\Longrightarrow\quad
\dot E=-2\kappa E.
$$

This is ordinary linear consensus. It verifies consistency of the proposed
conservation/dissipation pair, not a new graph-federated learning result.
Zero summary variance can also result from collapsed representations or
collapsed transport maps. Task-preservation constraints must be independently
measurable, rather than defined by the same energy they are meant to validate.

One possible motivation is consistent task-relevant responses under comparable
inputs, while preserving differences in class composition and useful local
propagation. Shared semantic probes, conditional flow responses, or other
observables could express this, but none is selected yet. Unequal responses
are not automatically harmful; unsupported classes must not be treated as
observed evidence, and evaluation labels cannot define a training target.

The two desired properties concern different dynamics:

- decrease of E_het along federated time tau;
- admissibility/stability of representation propagation along local time t.

Neither follows from the other. A dissipative correction in t can fail to
reduce cross-client disagreement in tau; averaging summaries can reduce
disagreement while hurting local prediction.

## Admissibility: preserve locality without assuming a model split

A compact high-level formulation is

$$
\min_{x_1,\ldots,x_M} E_{het}(x_1,\ldots,x_M)
\quad\text{subject to}\quad
K(x)=K_0,\quad x_m\in\mathcal A_m.
$$

Here A_m should eventually represent locally realizable dynamics and measurable
task-preservation/stability requirements. It must not be an arbitrary box
introduced just to create different client equilibria.

Conservation and admissibility must be enforced jointly. Independently clipping
each client's already balanced flux generally destroys its zero-sum property.
A feasible exchange can be zero if the constraints leave no descent direction;
strict energy descent is not guaranteed. Feasibility itself must be checked.

Nonincreasing energy bounded below implies convergence of energy values.
It does not alone imply state convergence, global optimality, or improved
accuracy. A minimum-heterogeneity equilibrium is a desired target. A first
theorem may only establish a stationary admissible state, a convex-case
minimizer under extra assumptions, or a bounded tracking error with local
learning and estimation disturbances.

## Realization: an exchange must change an actual graph system

The document's U_m = T_m^{-1}(J_m) is a conceptual placeholder. J_m is a
velocity of a summary in tau, whereas U_m is a modification of a graph field
in t. A compressed summary or transport need not be invertible. If knowledge
includes initial conditions, its realization may require an initial-state
change as well as a field change.

A later method needs a realization relationship, conceptually

$$
Obs_m(S_m^{new}) =
x_m+\Delta\tau J_m+\varepsilon_m^{real},
$$

in the chosen comparable coordinates. The realization error must be measured
from effective trajectories; it cannot be assumed zero because the requested
summary update was conservative. This is a necessary high-level link, not a
choice of generator injection or another specific implementation.

## Corrections to the earlier R003 reasoning

1. Native calibration does not mean there is no feedback. Training uses the
   injected forward, gradients change model parameters, and those parameters
   determine later native trajectories. The missing result is a bound on
   effective heterogeneity, not absence of a loop.
2. For a fixed basis/map and exact reduced linear dynamics, the old unprojected
   residual already gives A_eff=(1-beta)A_local+beta A_target. Proximal
   attraction alone is not a distinct new mechanism.
3. The factor (1-eta)^2 describes an ideal summary interpolation with a common
   eta, fixed target/coordinates, and actual realization of the interpolated
   state. It is not a bound between training rounds of the current nonlinear
   implementation.
4. Existing small injection ratios motivate investigation. They do not prove
   that small accuracy changes are caused by weak injection: accumulated
   propagation effects, gradients, alignment, and task sensitivity also matter.
5. A fitted reduced generator is not the full nonlinear Jacobian. Stability
   of a reduced operator or continuous-time field does not automatically imply
   stability of the full explicit Euler implementation.

## Next research output

Stay at the high level. Define one coherent tuple

$$
(\Xi,\;\text{comparability},\;K,\;E_{het},\;\mathcal A,\;\text{realization})
$$

with a motivating feature/structure example and a case in which naive
homogenization would be harmful. First justify why the conserved object helps
learning and why decreasing the energy removes a harmful difference.
Only then select a flux and implementation tools. R003 remains NEEDS_RESEARCH.
