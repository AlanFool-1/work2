# Current Handoff

Status: NEEDS_RESEARCH

Owner: research-agent (xzc, temporary research takeover explicitly requested by the user)

Updated: 2026-09-06 UTC

Handoff: R011

## Functional-map action-constraint completion — 2026-09-06 UTC

The human identified two flaws in R010: orthogonal operator novelty is not the
same as importing a capability the receiver lacks, and placing response
operators in a pre-shared coordinate system removes the substantive role of
the existing Functional Map machinery.

R011 replaces orthogonal mode transfer with transported finite-horizon action
constraints. Client `j` supplies small action pairs `Z_j -> P_j^tau(Z_j)`.
Functional Map `C_j_to_i` transports both sides to client `i`. The receiver's
knowledge defect is the failure of the diagram to commute:

`E_j_to_i = P_i^tau(C_j_to_i Z_j) - C_j_to_i P_j^tau(Z_j)`.

An external equation is complementary only when the map generalizes, the
held-out defect is nonzero, normal local backprop reduces that defect, and the
update is compatible with the receiver's task. Multiple sources contribute a
set of equations rather than modes for a server average. Already learned or
duplicate equations have negligible defect; harmful equations fail the task
gate.

The online path must remain light: reuse a low-dimensional trajectory basis,
descriptor, and Functional Map; exchange only a few `r x Q` action pairs at a
small set of horizons; realize them with an ordinary distillation loss. Do not
construct the full response Jacobian, compute receiver-specific orthogonal
subspaces, or run a matched local-only fork every round. Matched local-only is
an offline mechanism audit only.

Next action: implement the frozen CPU pilot in
`.collab/FUNCTIONAL_INTERTWINING_DYNAMICS.md` and
`.collab/functional_intertwining_pilot.yaml`. Start with a known ground-truth
map and three action classes: already satisfied, transferable missing, and
task harmful. Compare ground-truth, learned descriptor/cycle, wrong, and
identity maps. Fit maps and evaluate action defects on disjoint probes or
horizons to prevent the map from explaining away the signal.

Do not modify the production aggregator or launch graph training. Advance only
if map recovery/cycle consistency, held-out defect classification, receiver
distillation benefit, negative controls, and the cost reduction all pass.

## Historical R010: low-dimensional dynamics operator selection — 2026-09-06 UTC

The human rejected E015's retrospective flow decay as a useful next
experiment. That analysis observes old-controller magnitudes and cannot decide
what dynamical knowledge should be exchanged. Keep it only as historical side
evidence.

R010 proposed representation selection by comparing local dynamics
objects under matched rank and communication budgets. R010's leading candidate was
a low-rank finite-horizon causal response operator mapping shared port/time
perturbations to shared task-observation/time responses. Do not call it a
Hankel operator unless approximate lag stationarity is measured; the nonlinear
trajectory linearization can be time-varying.

Define complementarity as receiver-specific operator-subspace innovation, not
distance to a centroid. For source singular mode `b_jk` and receiver operator
subspace projector `P_i`, start from
`n_j_to_i_k = sigma_jk (I-P_i)b_jk`. Deduplicate innovations across sources by
incremental QR/SVD, then filter them by receiver realizability and gain over a
matched local-only update. The server forms a receiver-specific union of
nonredundant modes; it does not average models or publish a common prototype.

Historical action, superseded by R011: implement the measurement/evaluation pilot specified in
`.collab/LOW_DIMENSIONAL_DYNAMICS_OPERATOR_SELECTION.md` and
`.collab/operator_proxy_pilot.yaml`. Stage A must use known shared and
client-exclusive modes to test whether the proxies recover true
complementarity. Stage B uses frozen A-DGN checkpoints for `feature_shift`,
`structure_homophily`, and `mixed`, after the required synthetic artifacts
exist. Compare the current autonomous generator, affine generator,
delay-AR/Koopman, low-rank causal response operator, and raw-response upper
bound under matched budgets.

R010 did not authorize the 11-dataset matrix or a training-time aggregator.
Advance only if one proxy passes held-out response fidelity, missing-mode
recovery, bootstrap stability, and predicts realized benefit over local-only;
source/time-shuffled and redundant-source controls must fail. If the raw
response upper bound has no receiver benefit, revisit the port/observation
semantics or the transfer premise.

## Historical R009: marginal-gain exhaustion revision — 2026-09-06 UTC

The human supplied the learning-process premise now recorded at the top of
`methodv0.md`: a client should absorb a finite amount of useful complementary
knowledge and eventually become federatively self-sufficient. The equilibrium
criterion is no longer minimum response variance by itself. It is zero
incremental task gain from any receiver-realizable external response direction
relative to a compute- and norm-matched local-only update.

Treat the causal response kernel as the candidate knowledge interface and the
local response Jacobian as a realizability filter. Estimate predicted marginal
gain `G_i` on a training-internal guard split, then measure realized gain by
forking the same frozen checkpoint into local-only, true-external,
time/source-shuffled, and no-flow updates. Flow must be accepted because its
incremental gain is positive, and must stop when that gain is exhausted.

The old antisymmetric conservation constraint may remain as an optional
response-centering or numerical constraint. It is not a physical law of
learning because knowledge can be copied without depleting the sender.

This remains the learning-process hypothesis, but its direct early/middle/late
experiment is deferred until R010 selects a defensible low-dimensional proxy.

## Canonical method consolidation — 2026-09-06 UTC

`methodv0.md` now starts with an agent-transfer section that consolidates the
current method skeleton, the later causal-response-kernel refinement, the
joint realizable conservative solve, lifecycle placement, validation gates,
and unresolved choices. It is the canonical method specification; this file
remains the canonical operational handoff.

This consolidation does not advance the status or authorize production
implementation. The next action remains the frozen synthetic response and
joint-realizability gate below. The maximal-common-dynamics/sheaf proposal from
the later discussion is not yet an adopted part of Method v0.

## Literature and completed-check amendment — 2026-09-05 UTC

The user prioritizes Hamiltonian meta-learning and NeurIPS 2022 MP-NODE.
Primary-source findings and the newer NCF (ICLR 2025) comparison are in
`.collab/MULTISYSTEM_DYNAMICS_LITERATURE.md`. The causal kernel remains a
candidate interface, not an adopted new backbone. Compare it with
context-conditioned module fields and augmented message dynamics only where
they address a measured failure; do not stack all candidates or force a
Hamiltonian/shared-private parameterization.

Frozen random-model zero-port, causality and finite-difference checks, plus a
separate toy realization calculation, are now complete: see E013 and
`.collab/DYNAMICAL_RESPONSE_CHECKS.md`. Do not repeat these as if unperformed;
the remaining gate is trained-model coverage, common input/output semantics,
task relevance and actual graph-model realization. Class-conditioned sensors
in the toy are an unadopted alternative; missing classes and privacy remain
open. Use training-internal admissibility data, not test labels or the official
validation selection set as an optimization target.

Before implementation, specify static-output, time-shuffled, wrong-coupling
and no-transfer controls, communication cost and finite-update error budget.
Reconstructing self-generated hidden trajectories alone is not task evidence.
Matched continuous A and identity-centered P ridge are the same model; D017
corrects the earlier expectation that changing to P improves expressivity.
Source changes and graph training remain paused at NEEDS_RESEARCH.

## Goal and current scope

Refine and validate the exchanged dynamical knowledge before implementing the
conservative flow. The current strongest candidate is a finite-time causal
response kernel of the real ODE-GNN under common dynamical probes, rather than
an artificial concatenation of low-dimensional summaries.

Neither constraint is fundamental to the ODE viewpoint. A general native
weight matrix is an open candidate. No source change or new stability
architecture is requested at this handoff. `methodv0.md` is the current
research specification, and E009 provides the exchange-layer sanity result.
The existing generator fit is not yet trusted: its steady-state error ranges
from roughly `0.276` to `0.727` across the 11-dataset evidence. It is now a
compression diagnostic, not the definition of the public knowledge state.
The candidate kernel records input port, injection time `s`, observation time
`t`, output semantics, and response sign; parameter Jacobians are local
realization tools, not uploaded knowledge.

## Current evidence and interpretation

The existing low-rank generator/Functional Map/prototype implementation is
engineering-complete. Previous checks passed 18 unit tests and baseline
invariants. Completed single-seed runs show small injection and mixed gains,
but do not establish useful transfer or round-to-round heterogeneity descent.

Native calibration disables the current injection during measurement, but
injected training changes parameters and thereby subsequent native trajectories.
Feedback is therefore present. The old unprojected residual already gives an
idealized interpolation toward a target; renaming this a proximal step does
not supply a new mechanism.

The scalar change from native -2h to target -h requires correction +h, which
the dissipative projection deletes. Both resulting Euler maps are stable at
step size 0.1. This disproves necessity of correction dissipativity for stable
alignment; it does not establish an accuracy benefit from removing it.

## Research decision required

1. Freeze a small synthetic graph/model and define explicit dynamical ports:
   initial-state amplitude, neighbor coupling, and local damping. All ports
   must recover the native ODE exactly at zero amplitude.
2. Measure the causal response kernel from injection time `s` to observation
   time `t` using JVP/AD without constructing dense node Jacobians. Check the
   future-to-past causal mask and compare float64 AD against centered finite
   differences at several amplitudes.
3. Define `x_m` from the kernel blocks and output semantics, after transport;
   do not concatenate `z_m^0` and `A_m`. Record response coverage, nonlinear
   remainder, transport error, and task-observation support.
4. Compute the local realizable response subspace `D_m v_m` and solve the
   constrained convex exchange problem in the candidate research note. Check
   conservation in the ideal linearized system and drift after the actual
   finite parameter step.
5. Keep the ridge generator, discrete operator, and neural residual as
   compression diagnostics only. Do not implement conservative exchange until
   the kernel passes the measurement and realization gates.

## Invariants and implementation boundary

- Existing official A-DGN baseline, FedAvg lifecycle, persistent Adam, and
  paired best-validation evaluation remain unchanged.
- A future general-W model must be identified as a separate backbone variant.
  Existing module-off equivalence must remain reproducible.
- Keep research modules opt-in and do not upload node-level states or bases.
- Describe simulated aggregation as simulated, not cryptographically secure.
- Do not expand prototype engineering or tune injection strength for this task.

## Acceptance criteria for advancing the handoff

Advance to implementation when the causal response kernel is reproducible,
transportable, and locally realizable on the controlled synthetic benchmark.
The first implementation may exchange a small raw causal-kernel tensor;
operator compression is optional. Then run official baseline, no-flow,
unconstrained-flow, and constrained-flow comparisons with paired task metrics.

Do not select a barycenter, proximal controller, or complex W solely to
complete this handoff. The high-level causal argument is recorded in
`methodv0.md`; the stronger causal-kernel candidate is recorded in
`.collab/DYNAMICAL_RESPONSE_KERNEL_RESEARCH.md`. Source implementation remains
paused until the research gate passes.

## Evidence

- CONSERVATIVE_DYNAMICAL_KNOWLEDGE_FLOW_HIGH_LEVEL.md
- methodv0.md
- .collab/EXPERIMENTS.md: E003, E009, E010, E011, E012
- .collab/CONSERVATIVE_FLOW_REVIEW.md
- .collab/DECISIONS.md: D008-D011 (including scope corrections)
- .collab/EXPERIMENTS.md: E003, E006, E007
- Anti-SymmetricDGN-main/graph_heteropily/models/antisymmetric_dgn.py
- backbone_functional_dynamics_stable/models/s0/model.py
- backbone_functional_dynamics_stable/models/dissipative_injector.py
