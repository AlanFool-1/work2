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

Status: engineering-complete for the current specification and experimentally stable, but the transfer mechanism remains unvalidated.

## Established evidence

- Official 11-dataset baseline results are stored in `backbone_functional_dynamics_stable/BASELINE_RESULTS.json`.
- The controlled synthetic `feature_shift` baseline reaches `88.68 +/- 2.68%` paired test accuracy at seed 42 and 100 rounds; see `backbone_functional_dynamics_stable/SYNTHETIC_BENCHMARK.md`.
- A seed-42 multi-prototype Functional Dynamics matrix has complete results for all 11 real datasets across two run directories. Seven point estimates improve and four decline; mean and median deltas are `+0.3366` and `+0.0302` percentage points. Only four gains exceed `+0.5` points. See `.collab/EXPERIMENTS.md` entry E003.
- Current verification passes 18 unit tests plus all fast baseline invariants.

## Current empirical interpretation

- Functional Map optimization is finite, reduces its objective, and stays below the map condition limit.
- Consecutive trajectory bases are stable, but local generator fit error varies widely across datasets.
- Injection remains small at roughly 0.57%-1.32% of the native field in steady state.
- The dissipative spectral cap binds on most datasets, so raw correction magnitude is largely removed before injection.
- Prototype assignments never switch. With the configured margin of 1.0 and the current relative-improvement rule, the experiment behaves as fixed bootstrap clustering rather than adaptive multi-regime tracking.
- The single-seed gains are not sufficient to attribute improvement to cross-client dynamics transfer.

## Current open problems

1. Decide whether adaptive prototype switching is part of the intended mechanism; if so, replace the effectively frozen margin with a falsifiable setting.
2. Separate the effects of cross-client transfer, fixed clustering, Functional Map regularization, dissipative projection, and the small injection coefficient.
3. Define local-only, shuffled/wrong-prototype, single-prototype, and module-off matched controls.
4. Use controlled feature/structure synthetic regimes before interpreting real-dataset accuracy changes mechanistically.
5. Convert the research conclusion into one bounded, implementation-ready next task.

There is no active source-code task until the research agent changes `.collab/NEXT_TASK.md` to `READY_FOR_IMPLEMENTATION`.
