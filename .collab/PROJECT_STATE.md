# Current Project State

Updated: 2026-09-06 UTC

## Research objective

Functional-intertwining revision (2026-09-06 UTC): R011 replaces R010's
orthogonal response-mode novelty with transported finite-horizon action
constraints. Pairwise Functional Maps now have a central role: they transport
both an input function/probe and the source's evolved output. The receiver's
knowledge defect is the failure of evolve-after-transport and
transport-after-evolve to agree. An equation is complementary only when the
map generalizes, the held-out defect is nonzero, ordinary local backprop can
reduce it, and the induced update is task-compatible. Multiple sources add
constraints rather than forming a model/operator average. See
`.collab/FUNCTIONAL_INTERTWINING_DYNAMICS.md`.

The online design is lighter: periodic low-dimensional action sketches,
pairwise/cycle-consistent Functional Maps, and ordinary distillation replace
the full response Jacobian, response SVD, receiver-specific orthogonal search,
and per-round local-only counterfactual. R011 remains `NEEDS_RESEARCH`; the next
evidence is a frozen CPU known-map action-constraint pilot.

Historical R010 operator-selection revision (2026-09-06 UTC): the immediate bottleneck was no
longer whether the old correction norm decays, but which low-dimensional local
dynamics object can support interaction, composition, and receiver-specific
complementarity. R010 proposed comparing autonomous/affine generators, delay-AR/Koopman,
and a low-rank finite-horizon causal response operator under matched rank and
communication budgets. The response operator was R010's leading candidate because
its axes have shared port/time and task-observation/time semantics. Cross-client
composition was defined as a deduplicated union of modes missing from each
receiver, followed by realizability and marginal-gain gates. D022 supersedes
that orthogonal novelty definition. See
`.collab/LOW_DIMENSIONAL_DYNAMICS_OPERATOR_SELECTION.md`.

Learning-process revision (2026-09-06 UTC): the current research premise is
finite complementary knowledge absorption. A client's external dependency is
measured by the additional task gain of receiver-realizable causal-response
directions over a matched local-only update. Equilibrium means this marginal
gain is exhausted for every client; it does not require response consensus.
The conservative response constraint is secondary because learning copies
knowledge rather than transferring a depletable physical mass. R011 now tests
this premise through transported, unsatisfied action equations.

Method-transfer consolidation (2026-09-06 UTC): `methodv0.md` now contains a
front-loaded canonical summary of the latest causal-response-kernel candidate
and joint realizable conservative flow. The method specification and
operational handoff are no longer split across files for agent onboarding;
supporting `.collab` notes retain derivations and evidence. Research status
remains `NEEDS_RESEARCH`, and no source implementation or training is approved.

Latest literature/diagnostic amendment (2026-09-05 UTC): the user temporarily
assigns research to `xzc` and requires dynamics-grounded communication.
Hamiltonian meta-learning (ICLR 2021/2024), MP-NODE (NeurIPS 2022), and NCF
(ICLR 2025) are prioritized in `.collab/MULTISYSTEM_DYNAMICS_LITERATURE.md`.
No new backbone is adopted. E013 records completed random-model CPU response
checks; trained-task transfer and interface comparability remain unvalidated.
Matched A and identity-centered P ridge are equivalent (D017), correcting the
earlier expectation that P alone fixes the surrogate. That literature and
diagnostic pass is historical evidence under the current R011 gate.

This workspace studies heterogeneous graph federated learning through graph neural dynamics. For client `m`, feature heterogeneity enters primarily through the initial condition `H_m(0)`, while structural heterogeneity changes the graph propagation operator or vector field. The current direction augments FedAvg with a low-dimensional dynamics knowledge channel.

The active conceptual pipeline is now specified at Method v0 level around
Functional Map transport and dynamical constraint completion:

`Build local function spaces -> transport finite-time action equations -> measure held-out commutation defect -> absorb learnable task-compatible equations -> stop when useful defects are exhausted`

The current research object is a transported finite-time action equation. A
pairwise Functional Map carries both its input and evolved output between
client-local function spaces. Complementarity is an unsatisfied, learnable and
task-compatible equation, not an orthogonal mode or a shared/private parameter
split. Semantic probes, map identifiability, cycle consistency and local task
benefit still require controlled validation.

## Active code and workspace map

- `backbone_functional_dynamics_stable/`: current opt-in Functional Dynamics implementation and the primary implementation target.
- `backbone/`: official A-DGN + FedAvg baseline copy.
- `backbone_synthetic_csbm/`: controlled synthetic benchmark predecessor/reference.
- `Anti-SymmetricDGN-main/`: upstream A-DGN reference code.
- `datasets/`, `logs/`, and `checkpoints/`: generated data and experiment artifacts; intentionally excluded from Git.
- `CONSERVATIVE_DYNAMICAL_KNOWLEDGE_FLOW_HIGH_LEVEL.md`: current high-level research skeleton.
- `backbone_functional_dynamics_stable/FUNCTIONAL_DYNAMICS_IMPLEMENTATION.md`: active implementation notes and verification commands.

## Backbone and evaluation invariants

- Current implementation/comparison backbone: official A-DGN with 16 fixed Euler steps. General native weights are a research candidate, to be identified as a separate backbone variant if implemented.
- Federation: 10 persistent workers, global broadcast, full-model equal-weight FedAvg by default.
- Optimizer: persistent local Adam state.
- Evaluation: each client's best-validation round paired with that round's test metric, then mean and client standard deviation.
- Functional Dynamics is opt-in; disabling it must retain native-backbone behavior.
- The aggregation boundary is simulated secure aggregation, not a cryptographic protocol.

## Current Functional Dynamics implementation

The active implementation extracts low-rank trajectory bases and ridge-fitted generators from native A-DGN trajectories. Functional Maps translate client generators into canonical coordinates. The server aggregates canonical summaries, and clients pull shared dynamics back as a bounded low-rank correction.

Implemented experimental extensions include:

- descriptor bootstrap before dynamics alignment and injection;
- multiple dynamics prototypes selected from descriptor geometry;
- orthogonal or regularized Functional Maps;
- generator shape normalization with local speed restoration;
- diagnostics for generator fit, map residuals, basis staleness, correction stability, injection ratio, and cluster assignment.

Status: the existing Functional Dynamics implementation is engineering-complete and numerically stable, but its transfer mechanism is not validated. The ridge generator's steady-state relative fit error is roughly `0.276-0.727`, so it is not trusted as exchanged knowledge. R011 reuses the basis, descriptor and Functional Map machinery but replaces canonical averaging and generator pullback with pairwise transported finite-time action equations. E009 validates only the old exchange layer; the R011 map/action mechanism has not been run.

## Established evidence

- Official 11-dataset baseline results are stored in `backbone_functional_dynamics_stable/BASELINE_RESULTS.json`.
- The controlled synthetic `feature_shift` baseline reaches `88.68 +/- 2.68%` paired test accuracy at seed 42 and 100 rounds; see `backbone_functional_dynamics_stable/SYNTHETIC_BENCHMARK.md`.
- A seed-42 multi-prototype Functional Dynamics matrix has complete results for all 11 real datasets across two run directories. Seven point estimates improve and four decline; mean and median deltas are `+0.3366` and `+0.0302` percentage points. Only four gains exceed `+0.5` points. See `.collab/EXPERIMENTS.md` entry E003.
- Previous engineering verification passed 18 unit tests plus all fast baseline invariants; these were not rerun during the conceptual review.

## Current empirical interpretation

- Functional Map optimization is finite, reduces its objective, and stays below the map condition limit.
- Consecutive trajectory bases are stable, but local generator fit error varies widely across datasets.
- Injection remains small at roughly 0.57%-1.32% of the native field in steady state.
- The dissipative spectral cap binds on most datasets, so raw correction magnitude is largely removed before injection.
- Prototype assignments never switch. With the configured margin of 1.0 and the current relative-improvement rule, the experiment behaves as fixed bootstrap clustering rather than adaptive multi-regime tracking.
- The single-seed gains are not sufficient to attribute improvement to cross-client dynamics transfer.

## Current open problems

1. Establish that descriptor/cycle information identifies a meaningful Functional Map without using the same action data later scored as knowledge defect.
2. Show that held-out commutation defect distinguishes learned, transferable-missing, and task-harmful actions.
3. Determine the smallest action probe/horizon bank that covers feature, structure and joint heterogeneity.
4. Keep native antisymmetry and correction dissipativity optional. A general graph vector field is compatible with the ODE view; finite-horizon numerical and gradient behavior is a separate question, with no mandatory trust-region architecture yet.
5. Verify that ordinary action-pair distillation reduces held-out defect and improves the receiver task without a response parameter Jacobian.
6. Validate cycle consistency and control wrong/no-map, shuffled, repeated, and task-harmful equations.
7. Measure communication and wall time against the full response-Jacobian R010 path.
8. Implement training-time exchange only after these gates, without changing official A-DGN/FedAvg behavior.

The active research handoff is recorded in `.collab/NEXT_TASK.md`.
