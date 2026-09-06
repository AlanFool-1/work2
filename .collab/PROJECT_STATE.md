# Current Project State

Updated: 2026-09-06 UTC

## Research objective

Operator-selection revision (2026-09-06 UTC): the immediate bottleneck is no
longer whether the old correction norm decays, but which low-dimensional local
dynamics object can support interaction, composition, and receiver-specific
complementarity. R010 compares autonomous/affine generators, delay-AR/Koopman,
and a low-rank finite-horizon causal response operator under matched rank and
communication budgets. The response operator is the leading candidate because
its axes have shared port/time and task-observation/time semantics. Cross-client
composition is defined as a deduplicated union of modes missing from each
receiver, followed by realizability and marginal-gain gates. This is a research
hypothesis, not an adopted algorithm. See
`.collab/LOW_DIMENSIONAL_DYNAMICS_OPERATOR_SELECTION.md`.

Learning-process revision (2026-09-06 UTC): the current research premise is
finite complementary knowledge absorption. A client's external dependency is
measured by the additional task gain of receiver-realizable causal-response
directions over a matched local-only update. Equilibrium means this marginal
gain is exhausted for every client; it does not require response consensus.
The conservative response constraint is secondary because learning copies
knowledge rather than transferring a depletable physical mass. R010 remains
`NEEDS_RESEARCH`; the next evidence is an operator-identification pilot with
known complement modes, followed by frozen synthetic checkpoints.

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
diagnostic pass is historical evidence under the current R010 gate.

This workspace studies heterogeneous graph federated learning through graph neural dynamics. For client `m`, feature heterogeneity enters primarily through the initial condition `H_m(0)`, while structural heterogeneity changes the graph propagation operator or vector field. The current direction augments FedAvg with a low-dimensional dynamics knowledge channel.

The active conceptual pipeline is now specified at Method v0 level around
task-aware conservative dynamical knowledge flow:

`Define task-relevant dynamical knowledge -> Establish comparability -> Exchange under conservation and local admissibility -> Assess harmful heterogeneity and task behavior`

The current research object is a transported, task-relevant dynamical response
state. Semantic probes, Functional Map details, and local realization are
implementation choices still requiring controlled validation. Proximal targets,
explicit shared/private decomposition, anti-symmetric native weights, and
correction dissipativity are not high-level requirements.

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

Status: the existing Functional Dynamics implementation is engineering-complete and numerically stable, but its transfer mechanism is not validated. The current bottleneck is the exchanged knowledge definition: the ridge generator's steady-state relative fit error is roughly `0.276-0.727` across the 11-dataset evidence, so it is not trusted as the public state. Method v0 now points toward finite-time responses to common public probes after transport; the leading candidate is a low-rank finite-horizon causal response operator indexed by dynamical port and injection/observation time. Generators, delay operators, and the raw response are matched candidates or upper bounds in R010. E009 validates the old exchange layer only.

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

1. Define task-relevant dynamical knowledge and explain what conservation preserves; shared semantic probes are one candidate.
2. Define harmful disagreement and locally admissible task behavior without presupposing an explicit shared/private split.
3. Derive a compatible conservative flux and heterogeneity energy, stating the assumptions and time scale of descent or convergence claims.
4. Keep native antisymmetry and correction dissipativity optional. A general graph vector field is compatible with the ODE view; finite-horizon numerical and gradient behavior is a separate question, with no mandatory trust-region architecture yet.
5. Explain how exchange is realized in local models and distinguish exchange from local-learning source terms. Injected training already influences later native calibration through learned parameters; disabling correction during measurement does not remove all feedback.
6. Select a low-dimensional proxy using held-out response prediction, known missing-mode recovery, subspace stability, and receiver benefit.
7. Validate causal response-operator measurement, transport, and local realization on frozen trained models.
8. Implement any training-time exchange only after these gates, without changing official A-DGN/FedAvg behavior.

The active research handoff is recorded in `.collab/NEXT_TASK.md`.
