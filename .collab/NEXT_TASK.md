# Current Handoff

Status: NEEDS_RESEARCH

Owner: research-agent (`root`)

Updated: 2026-09-05 UTC

Handoff: R001

## Goal

Audit the current multi-prototype Functional Dynamics idea against the completed 11-dataset results and its mechanism diagnostics, then produce a falsifiable improvement or a justified decision to simplify/abandon the branch.

## Problem

The current method has a complete implementation and produces valid experiment artifacts, but its seed-42 performance changes relative to the official baseline are mixed. Improvements appear on Cora, CiteSeer, Computers, and Photo; several other datasets are nearly unchanged or slightly worse. Final accuracy/AUC alone does not establish that functional alignment and prototype sharing are responsible for the gains.

## Evidence

- Summary and baseline comparison: `.collab/EXPERIMENTS.md`, E003.
- Baselines: `backbone_functional_dynamics_stable/BASELINE_RESULTS.json`.
- First four result directories: `logs/{Cora,CiteSeer,PubMed,Computers}_disjoint/clients_10/*functional_dynamics_multi_proto_all11_20260905_013318*/`.
- Remaining seven result directories: paths listed in `backbone_functional_dynamics_stable/run_logs/functional_dynamics_multi_proto_all11_20260905_023525/status.tsv`.
- Each completed result directory contains `functional_dynamics.csv`, cluster plots, canonical generator plots, `client_best.csv`, `config.json`, and `result.json`.
- The first orchestration status labels Computers as failed despite a complete `result.json`; treat this as an orchestration-record inconsistency, not evidence of an algorithm failure, until inspected.

## Engineering verification already available

- The implementation is opt-in and includes native-path equivalence tests.
- Result artifacts exist for all 11 real datasets.
- This workspace initialization did not rerun the model tests or experiments.

## Research questions

1. Do generator-fit error, descriptor/dynamics residuals, injection-to-native norm ratio, and basis staleness predict per-client or per-dataset performance changes?
2. Are prototype assignments stable and meaningfully related to graph/feature heterogeneity, or are they an extra degree of freedom without explanatory value?
3. Does the shared correction add information beyond a matched local-only or shuffled-prototype control?
4. Should the next design retain multi-prototype Functional Maps, simplify to a lower-variance mechanism, or return to a feature-initial-state-only intervention?

## Invariants for any next design

- Preserve official A-DGN, FedAvg, persistent Adam, and paired best-validation evaluation.
- Keep the new mechanism opt-in and preserve exact module-off equivalence.
- Do not describe simulated aggregation as cryptographically secure.
- Avoid reintroducing an abandoned idea unless `.collab/DECISIONS.md` records new evidence that resolves the earlier failure.

## Required research output

1. Add a compact diagnostic analysis to `.collab/EXPERIMENTS.md` with evidence paths.
2. Record adopted and abandoned directions in `.collab/DECISIONS.md`.
3. Update `.collab/PROJECT_STATE.md` if the global idea changes.
4. Replace this handoff with one implementation-ready specification and set `Status: READY_FOR_IMPLEMENTATION`, or document why no code change is justified.
5. Append a short entry to `.collab/logs/research.md`.

## Acceptance criteria for the next handoff

- The proposed mechanism is stated as a testable cause-and-effect claim.
- The specification names affected modules and preserved invariants.
- Required diagnostics and matched controls can distinguish success from leakage or a no-op.
- The implementation agent can proceed without asking the human to reconstruct the research conversation.
