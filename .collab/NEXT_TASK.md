# Current Handoff

Status: NEEDS_RESEARCH

Owner: research-agent (xzc, temporary research takeover explicitly requested by the user)

Updated: 2026-09-06 UTC

Handoff: R008

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
