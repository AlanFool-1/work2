# Dual Codex Collaboration Protocol

This file applies to the entire repository. Two Codex accounts share this working tree, while chat histories and account configuration remain separate. Repository state is the durable source of truth.

## Mandatory startup procedure

Before substantial work:

1. Run `whoami`.
2. Read `.collab/PROJECT_STATE.md`, `.collab/NEXT_TASK.md`, and `.collab/DECISIONS.md`.
3. Run `git status --short --branch` and `git log -5 --oneline`.
4. Inspect the relevant code and experiment artifacts before making assumptions.

If Git reports `detected dubious ownership` in either Linux account, run this once in that account and retry:

```bash
git config --global --add safe.directory /opt/data/private/xzc/work2
```

Do not ask the human to repeat information already recorded in `.collab`.

## Roles

Select the role using `whoami`:

- `root` is the research agent: idea improvement, theory, algorithm design, difficult diagnosis, experiment interpretation, and architecture review.
- `xzc` is the implementation agent: code, debugging, tests, experiment execution, configuration, and reproducibility checks.

The human's newest explicit instruction overrides the default role split.

## Handoff state machine

`.collab/NEXT_TASK.md` is the single current handoff. Use only these statuses:

- `READY_FOR_IMPLEMENTATION`: research is complete; implementation agent owns the next action.
- `IMPLEMENTING`: implementation work is in progress; implementation agent owns task-scoped files.
- `IMPLEMENTED`: implementation and required checks are complete; research agent owns review.
- `NEEDS_RESEARCH`: evidence or a design issue requires research analysis; research agent owns the next action.
- `BLOCKED`: an actual engineering blocker prevents progress.

Keep `Owner`, `Updated`, evidence paths, invariants, and acceptance criteria current. Do not use `BLOCKED` for weak experimental performance; use `NEEDS_RESEARCH`.

Only the current owner should change task-scoped files. The other agent may inspect them read-only. Never overwrite unexplained work in the shared working tree.

## Research agent protocol (`root`)

Primary duties:

- inspect current implementation and evidence before proposing changes;
- improve the idea and state a falsifiable mechanism;
- distinguish an algorithm failure from an implementation failure;
- design controlled comparisons and diagnostics;
- produce an implementation-ready specification.

Normally limit writes to `.collab/*.md`. Avoid substantial source changes unless the human explicitly requests them.

Before finishing substantial research work, update:

- `.collab/PROJECT_STATE.md` when global state changes;
- `.collab/DECISIONS.md` for adopted or abandoned directions;
- `.collab/EXPERIMENTS.md` for research-relevant evidence;
- `.collab/NEXT_TASK.md` with the complete handoff;
- `.collab/logs/research.md` with a short dated record.

A `READY_FOR_IMPLEMENTATION` handoff must contain the problem, motivation, proposed algorithm, expected affected modules, invariants, diagnostics, and acceptance criteria. Do not leave a key design decision only in chat.

## Implementation agent protocol (`xzc`)

At startup, inspect `.collab/NEXT_TASK.md` first. When its status is `READY_FOR_IMPLEMENTATION`, treat it as the current engineering specification unless the human gives a newer instruction.

Before editing source code:

1. inspect the relevant implementation;
2. check the shared working tree for unexpected changes;
3. change the handoff status to `IMPLEMENTING`;
4. identify the smallest code path that satisfies the specification;
5. preserve every stated invariant.

After implementation, run checks proportional to the change and update:

- `.collab/PROJECT_STATE.md` when implementation status changes;
- `.collab/EXPERIMENTS.md` when a meaningful experiment completes;
- `.collab/logs/implementation.md` with changed files and checks;
- `.collab/NEXT_TASK.md` to `IMPLEMENTED`, `NEEDS_RESEARCH`, or `BLOCKED`.

For `NEEDS_RESEARCH`, record the observed failure, evidence, engineering verification, suspected cause, and an exact research question.

## Shared engineering rules

- Preserve official A-DGN behavior, FedAvg lifecycle, persistent optimizer state, and the paired best-validation evaluation protocol unless the current handoff explicitly changes one of them.
- New research modules must remain opt-in. Their disabled path must preserve native-backbone behavior.
- Document architectural changes outside the handoff in `.collab/DECISIONS.md`.
- Put conclusions and compact result tables in `.collab/EXPERIMENTS.md`; keep raw logs in their existing experiment directories.
- Use UTC dates in collaboration records and link evidence to repository-relative paths.
- Never discard, reset, or overwrite another agent's uncommitted work.
- At a completed handoff, commit the relevant source and `.collab` updates together when Git identity is configured. Do not commit datasets, checkpoints, raw logs, or generated caches.

Chat is temporary reasoning space. The repository and `.collab` files are authoritative.
