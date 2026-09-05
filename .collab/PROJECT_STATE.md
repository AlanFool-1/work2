# Current Project State

Updated: 2026-09-05 UTC

## Research objective

This workspace studies heterogeneous graph federated learning through graph neural dynamics. For client `m`, feature heterogeneity enters primarily through the initial condition `H_m(0)`, while structural heterogeneity changes the graph propagation operator or vector field. The current direction augments FedAvg with a low-dimensional dynamics knowledge channel.

The active conceptual pipeline is:

`Extract local dynamics -> Translate to a canonical functional space -> Share prototypes -> Complement native dynamics`

## Active code and workspace map

- `backbone_functional_dynamics_stable/`: current opt-in Functional Dynamics implementation and the primary implementation target.
- `backbone/`: official A-DGN + FedAvg baseline copy.
- `backbone_synthetic_csbm/`: controlled synthetic benchmark predecessor/reference.
- `Anti-SymmetricDGN-main/`: upstream A-DGN reference code.
- `datasets/`, `logs/`, and `checkpoints/`: generated data and experiment artifacts; intentionally excluded from Git.
- `FUNCTIONAL_DYNAMICS_HIGH_LEVEL_FOR_CODEX.md`: high-level algorithm context.
- `backbone_functional_dynamics_stable/FUNCTIONAL_DYNAMICS_IMPLEMENTATION.md`: active implementation notes and verification commands.

## Backbone and evaluation invariants

- Model backbone: official A-DGN with 16 fixed Euler steps.
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

Status: experimental and awaiting research review of the latest 11-dataset evidence.

## Established evidence

- Official 11-dataset baseline results are stored in `backbone_functional_dynamics_stable/BASELINE_RESULTS.json`.
- The controlled synthetic `feature_shift` baseline reaches `88.68 +/- 2.68%` paired test accuracy at seed 42 and 100 rounds; see `backbone_functional_dynamics_stable/SYNTHETIC_BENCHMARK.md`.
- A seed-42 multi-prototype Functional Dynamics matrix has results for all 11 real datasets across two run directories. Accuracy/AUC changes are mixed and modest overall; see `.collab/EXPERIMENTS.md` entry E003.

## Current open problems

1. Determine whether the modest, nonuniform gains are explained by useful dynamics transfer, weak injection, unstable alignment, or prototype assignments.
2. Analyze the existing Functional Dynamics diagnostics rather than relying only on final accuracy/AUC.
3. Define matched controls and ablations that isolate the claimed mechanism.
4. Convert the research conclusion into one bounded, implementation-ready next task.

There is no active source-code task until the research agent changes `.collab/NEXT_TASK.md` to `READY_FOR_IMPLEMENTATION`.
