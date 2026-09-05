# Design Decisions

Updated: 2026-09-05 UTC

## D001 - Fixed collaboration roles

Decision:

- `root` is the research agent responsible for idea improvement, theory, diagnosis, experiment interpretation, and implementation-ready specifications.
- `xzc` is the implementation agent responsible for code, debugging, tests, and experiments.

Reason:

The two accounts do not share chat context. Fixed ownership and file-based handoffs prevent the human from becoming a message relay.

Status: Active.

## D002 - Repository state is the shared memory

Decision:

Use `AGENTS.md` plus `.collab/` as the durable protocol. Keep only current state, decisions, tasks, and research-relevant evidence; do not store full chat transcripts.

Reason:

These files are versionable, searchable, and recoverable by either account in a fresh session.

Status: Active.

## D003 - Preserve the official training and evaluation backbone

Decision:

Research additions remain opt-in and preserve official A-DGN, the FedAvg lifecycle, persistent optimizer state, and per-client best-validation paired test evaluation unless a later handoff explicitly changes an invariant.

Reason:

This keeps comparisons attributable to the research mechanism.

Status: Active.

## D004 - Functional Dynamics remains experimental

Decision:

Treat the current multi-prototype Functional Dynamics implementation as a candidate mechanism rather than an accepted final method.

Reason:

The seed-42 11-dataset matrix shows modest, nonuniform changes and lacks the matched controls needed to establish mechanism validity. Diagnostics also show zero prototype switches under margin 1.0, small effective injection, and frequent spectral clipping.

Status: Under review by handoff R001.

## D005 - Start with a file protocol, without automatic watchers

Decision:

Use manual account startup plus automatic reading of repository instructions. Do not add an `inotify`/`codex exec` relay during the first phase.

Reason:

The handoff format and concurrent-edit discipline should stabilize before unattended execution is introduced.

Status: Active.
