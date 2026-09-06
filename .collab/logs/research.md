# Research Agent Log

## 2026-09-06 UTC — R011 Functional Map action constraints

The human identified that R010 still resembled a personalized orthogonal
shared/private construction: subspace novelty is not equivalent to importing a
missing capability, and common response coordinates bypassed the existing
Functional Map. Replaced it with a commuting-diagram view.

Each client now exposes a few finite-horizon action pairs in its local function
space. A pairwise Functional Map transports both input and evolved output. The
receiver's knowledge defect is evolve-after-transport minus
transport-after-evolve. Complementary knowledge is an independently
transportable, currently unsatisfied, locally learnable, task-compatible
equation. Multi-client collaboration accumulates equations instead of averaging
operators or collecting orthogonal modes.

The new online path removes full response Jacobians, response SVD,
per-receiver subspace search, and per-round local-only counterfactuals. Recorded
D022, E017, R011, the formal design and a validated pilot manifest. No numeric
pilot, production edit, or training run was performed.

## 2026-09-06 UTC — R010 low-dimensional operator-selection redesign

The human rejected the retrospective decay of old correction magnitudes as an
uninformative next experiment and redirected the research to the local
dynamics object itself. Reframed the gate around held-out dynamics fidelity,
cross-client semantic comparability, receiver-specific complement recovery,
and local absorbability.

The leading candidate is a low-rank finite-horizon causal response operator,
with block-Hankel structure treated as an empirical special case rather than an
assumption for the nonlinear ODE-GNN. Cross-client combination uses a
deduplicated union of source modes outside the receiver's current operator
subspace, followed by realizability and marginal-gain gates. Recorded the
formal design and falsification rules in
`.collab/LOW_DIMENSIONAL_DYNAMICS_OPERATOR_SELECTION.md`, preregistered R010 in
`.collab/operator_proxy_pilot.yaml`, and updated Method v0, D021, E016, project
state, weekly report, and handoff. No new training or numeric experiment was
run.

## 2026-09-06 UTC — finite complementary knowledge absorption premise

The human clarified the learning-process motivation for knowledge flow: FL
should progressively reduce each client's dependence on useful external
knowledge until additional collaboration has no marginal value. Revised the
canonical Method v0 entry around external marginal gain over a matched
local-only update and defined federated self-sufficiency by gain exhaustion.

This changes the status of the old conservation argument. Knowledge is
non-rival, so antisymmetric response flow can be a centering or numerical
constraint but is not the fundamental reason exchange stops. The causal
response kernel remains the external knowledge interface; receiver
realizability and task gain decide whether it is absorbed.

Prepared R009 and a weekly report draft. Added a retrospective analysis of the
completed 11-dataset logs: the raw correction norm declines from the first to
last active 20% on 11/11 datasets (median late/early ratio 0.427), and the
injected/native ratio declines on 10/11 (median ratio 0.730), but the
post-projection safe norm declines on only 1/11 (median ratio 1.189). These are
old-method proxies, not measurements of marginal external information gain.

## 2026-09-06 UTC — canonical Method v0 agent-transfer summary

Audited `methodv0.md` against the newer causal-response-kernel research note,
R008, D016-D018, and the completed CPU checks. The latest design had been split:
the method body contained public probe responses and sequential flow/realization,
while the newer note defined port-time causal kernels and a joint realizable
conservative solve.

Added a front-loaded canonical transfer section to `methodv0.md` containing the
current exchanged object, optimization, one-round lifecycle, fixed/open choices,
validation thresholds, and explicit implementation boundary. Updated D019,
PROJECT_STATE, and R008 pointers. Status remains `NEEDS_RESEARCH`; no production
source, training, checkpoints, or experiment artifacts changed.

## 2026-09-05 UTC — temporary takeover, literature and completed CPU diagnostics

Actual account is `xzc`; the human explicitly requests temporary research
takeover. Inspected shared state, Method v0, native A-DGN/Functional Dynamics
and existing evidence. Shared R007/R008 updates arrived during this work;
preserved and amended the latest handoff rather than reverting them.

Studied Hamiltonian meta-learning, MP-NODE and NCF/CoDA/GG-ODE/GREAT/LEADS.
Recorded primary sources, boundaries and project inferences in
`MULTISYSTEM_DYNAMICS_LITERATURE.md`. No final new backbone selected.
The candidate native causal-response and realizable-flow derivation is in
`DYNAMICAL_RESPONSE_KERNEL_RESEARCH.md`; reproducible CPU checks are in
`DYNAMICAL_RESPONSE_CHECKS.md`. They confirm matched A/P ridge equivalence,
native causal derivatives and toy infinitesimal conservation, not task transfer.

Updated D017-D018, E013-E014, global state and the R008 research amendment.
Communication must remain dynamics-grounded. No production source, Method v0,
training or checkpoints changed. Git identity is unset, so no commit was made.

## 2026-09-05 UTC — causal response-kernel refinement

A concurrent research note in the shared workspace refined the common-probe
idea into a finite-time causal response kernel. The exchanged object is the
response of the real graph ODE/Euler system to a defined dynamical port applied
at time `s`, observed at time `t`, with port and output semantics retained.

This is stronger than using a raw probe trajectory or `vec(z) || vec(A)`: it
has a direct intervention/response meaning and does not require an autonomous
low-dimensional generator to close the projected dynamics. Parameter Jacobians
are local realization tools only. The next research check is a frozen
synthetic AD/finite-difference oracle with causality, nonlinear remainder,
coverage, and finite-step conservation diagnostics.

Updated D016, E012, PROJECT_STATE, and R008. Did not modify the concurrent
`.collab/DYNAMICAL_RESPONSE_KERNEL_RESEARCH.md` note or source code.

## 2026-09-05 UTC — redefine exchanged knowledge by common-probe response

The user identified that `vec(z_m^0) || vec(A_m)` is still an artificial state
and can collapse the contribution into ordinary consensus. Updated Method v0
so the exchanged state is the finite-time response of every client graph
dynamical system to the same public probe bank, after Functional Map transport:
\(x_m=\operatorname{vec}\{R_{m,q}^{\ell}-R_{m,q}^{0}\}_{q,\ell}\).

The generator \(A_m\) is now only a compression diagnostic. A high current
generator error (`0.276-0.727` steady-state medians) does not invalidate direct
probe-response exchange, but it prevents replacing those responses with a
generator without held-out rollout evidence. Local realization is defined by
the Jacobian of the probe-response map with respect to low-rank field
parameters, followed by a constrained convex quadratic solve and actual
post-update response measurement.

Updated D015, E011, PROJECT_STATE, and R007. No source code or graph training
run was changed.

## 2026-09-05 UTC — operator identification bottleneck

The user identified the central risk in Method v0: the fitted low-dimensional
generator may have too much error to represent transferable knowledge. Audited
the current implementation and E003 evidence. The logged `dyn_fit_error` is an
in-window one-step velocity residual, with steady-state medians roughly
`0.276-0.727` across the 11 datasets; no held-out time-block or multi-step
validation is currently logged.

Updated `methodv0.md` to add an operator validation gate. The first comparison
is continuous ridge generator versus directly fitted discrete propagator
`z_(t+1)=P z_t`, plus an affine variant. These are convex ridge problems and
avoid amplifying finite-difference noise by dividing by the Euler step size.
A small neural residual remains a later nonconvex diagnostic, not the default
knowledge object. Reinforced the prohibition on SVD for large graph operators.

Updated D014, E010, PROJECT_STATE, and R006. Conservative-flow source
implementation is paused until an operator passes time-block and rollout
validation.

## 2026-09-05 UTC — v0 concretization and toy exchange result

Refined `methodv0.md` into a minimal experimental specification. The public
state is one transported response vector containing initial-state and fitted
finite-time vector-field responses. Exchange uses a convex quadratic objective
over pairwise antisymmetric edge flows, with symmetric nonnegative gates for
local admissibility. FedAvg sample weights are separated from dynamical-state
weights and must be controlled rather than assumed valid.

Added a numerical policy forbidding SVD on large graph Laplacians, adjacency
matrices, or node-level feature matrices. v0 uses trajectory QR and ridge
linear solves; legacy low-dimensional Functional Map SVD remains a capped
historical utility to replace or isolate.

Ran a deterministic 10-client, 6-dimensional toy exchange. With
`delta_tau=0.25` and `kappa=0.4`, full connectivity reduced heterogeneity
energy from `4.69634122` to `0.06941607` (ratio `0.01478088`) in 20 monotone
steps, matching `0.9^40`. Two admissible groups reduced it monotonically to
`3.28964478` (ratio `0.70046971`) while retaining cross-group difference.
Maximum weighted conservation residual was below `8e-17` in both cases.

Updated D013, E009, and prepared R005 for minimal implementation. No source
code or training experiment was changed.

## 2026-09-05 UTC — Method v0 conservative dynamical knowledge-flow specification

Responding to the research direction, wrote `methodv0.md` in Chinese as the
current high-level method specification. Recentered the paper story on private
heterogeneous graph dynamical systems, transport of task-relevant dynamical
responses, conservative exchange flux, heterogeneity-energy dissipation, and
an admissible non-necessarily-consensus equilibrium.

Clarified that the weighted first moment is conserved only during a frozen
exchange phase; local learning is a separate source term. Added a task-aware
observation operator after Functional Map transport, pairwise antisymmetric
fluxes to preserve conservation under local constraints, an explicit local
realization residual, two time scales, falsifiable predictions, diagnostics,
and matched controls. Proximal consensus, prototypes, correction dissipativity,
and native anti-symmetry are no longer the high-level motivation.

Updated D012, E008, PROJECT_STATE, and R004. Source code and experiments were
not changed or run.

## 2026-09-05 UTC — native W need not inherit A-DGN constraints

Rechecked the upstream field, injected forward path, and correction projection.
The user's shallow-dynamics motivation does not require anti-symmetric native
weights or correction-only dissipativity. Recorded a scalar counterexample:
moving from -2h to -h needs +h although both resulting Euler maps are stable
at step size 0.1. General W is an open backbone candidate, not an implemented
change or a claimed accuracy improvement.

Clarified that low-rank Euler norms are surrogates and that trust regions,
backtracking, shared semantic probes, and proximal targets are not fixed
requirements of the high-level skeleton. Existing injected training already
affects subsequent native calibration through parameters. Updated D008-D011,
E006/E007, global state, and R003. Status remains NEEDS_RESEARCH; source code
was not changed and experiments/tests were not run in this review.

## 2026-09-05 UTC

Initialized the dual-Codex shared state from the existing Functional Dynamics design, implementation notes, baselines, and latest real-dataset run artifacts.

Main conclusion:

The current multi-prototype method is implemented and produces complete diagnostics, but its initial 11-dataset performance is mixed and does not yet validate the proposed transfer mechanism.

Updated:

- `.collab/PROJECT_STATE.md`

## 2026-09-05 UTC — redesign requested by human

The next step is explicitly not prototype tuning. The current intervention is
too weak and has no explicit contraction objective for cross-client dynamics.
The research direction is changed to a consensus-control formulation:

1. define a heterogeneity energy over effective initial-state and vector-field
   summaries;
2. construct a canonical barycentric target rather than treating prototypes as
   the main mechanism;
3. apply a proximal feedback update that contracts disagreement while retaining
   a client-native residual;
4. project against stability of the total effective field, since projecting the
   correction alone can discard the symmetric component needed to reduce
   disagreement;
5. close the loop by measuring the effective controlled dynamics on later
   rounds without circularly using the correction to define its own target.

R002 matched-control implementation is superseded pending this redesign.

Candidate insight recorded as D008: use a proximal dynamical-consensus update
over canonical initial-state descriptor, generator shape, and log-speed. A
fixed-target common coupling contracts weighted disagreement by a known factor;
target drift and estimation/map errors can be logged as residual terms. The
stability check must apply to the total effective generator. This is still a
research hypothesis and is intentionally not marked ready for implementation.
- `.collab/NEXT_TASK.md`
- `.collab/DECISIONS.md`
- `.collab/EXPERIMENTS.md`

Next owner: research agent (`root`), handoff R001.

## 2026-09-05 UTC — conservative-flow skeleton review

Reviewed `CONSERVATIVE_DYNAMICAL_KNOWLEDGE_FLOW_HIGH_LEVEL.md`. Accepted its
high-level structure as the project research direction, while refining the
central insight: the objective is not full consensus of all client dynamics.
Only transferable, task-harmful disagreement measured on shared semantic
probes should be dissipated; an admissible client-native residual must remain.

The equation `sum_m p_m J_m = 0` should be presented as conservation of the
exchange flux, not as a physical conservation law. A raw-summary variance is
insufficient because it can reward destructive homogenization. Before coding,
derive a task-aware transported energy, a conservative flux, a total-field
stability condition, and non-circular effective-trajectory timing.

Updated D009, E005, PROJECT_STATE, and retained `NEXT_TASK.md` at
`NEEDS_RESEARCH`.

## 2026-09-05 UTC — dissipativity distinction

Reviewed whether the code's correction dissipativity is necessary. Conclusion:
it is not a high-level requirement. A-DGN's anti-symmetric parameterization is
the backbone's design; the new method needs federated-time heterogeneity-energy
dissipation. For finite fixed-step rollouts, use a bounded-growth/trust-region
condition on the total effective field or Euler map. The current correction-only
projection may delete the consensus direction while failing to certify the
nonlinear total field. Recorded as D010 and E006; implementation remains paused.

## 2026-09-05 UTC — R002 handoff

Reviewed the consolidated diagnostics. The current method is a conservative
low-rank complement: effective injection is about 0.57%-1.32% of the native
field, spectral projection is active on most datasets, and no prototype
switches occur under the configured margin. Small metric deltas are therefore
expected, but they do not identify whether the transferred direction is useful.

Decision:

- keep the default beta, spectral cap, and switch margin unchanged;
- require matched module-off, local-only/no-transfer, single-prototype, full,
  and wrong-prototype controls;
- test the controls first on Synthetic feature, structure, and mixed regimes.

Updated:

- `.collab/DECISIONS.md` (D006)
- `.collab/NEXT_TASK.md` (R002, READY_FOR_IMPLEMENTATION)
- `.collab/PROJECT_STATE.md`
## 2026-09-06 UTC — R012 removes FedAvg from the candidate method

The human identified that retaining FedAvg would leave parameter averaging as
the dominant explanation for cross-client learning and convergence. Revised
the current method into a no-aggregation Functional Action Flow. Clients keep
persistent local parameters; the server routes Functional Map transported
action pairs; receiver updates are generated only by task-compatible
intertwining-residual gradients. Effective flow vanishes when an equation is
learned, rejected, or locally unrealizable. FedAvg is now a baseline only.

Updated D023, E018, PROJECT_STATE, NEXT_TASK, the R011/R012 method note,
`methodv0.md`, the weekly report, and the pilot manifest. No source code,
checkpoint, dataset, or numeric experiment was changed.
