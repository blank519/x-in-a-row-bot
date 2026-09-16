# Orca common pipeline mechanics

This file owns harness mechanics shared by code and experiment tickets. If a
command is rejected, load the version-matched orchestration guide as directed by
the installed `orchestration` skill; do not guess flags.

## Setup

- Confirm the Orca runtime and experimental orchestration feature.
- Default workers to `--agent pi` and `--worktree current`.
- Read `max_iterations` from the ticket; default 3.
- Create a Run with an objective containing ticket title and type.

## Task specs

Build specs from the appropriate complete `.orca/roles/<role>.md`, ticket/plan
unit, and concise retry feedback. Role files direct workers to the required
project playbooks. Create dependency edges from planner to implementers and from
all implementers to the batch evaluator.

## Dispatch and fan-out

Create all dispatchable tasks, but start no more than `max_parallel_workers`
(default 2). Only units explicitly marked independent may overlap. As workers
settle, release them and start queued work until all required units finish.

## Waiting protocol

Use blocking rolling waits, never polling worker terminals:

```text
orca orchestration check --wait --types worker_done,escalation,question --timeout-ms <ms> --json
```

Process every message in a Delivery:

- reply to each question;
- capture each `worker_done` body;
- release each settled worker using its dispatch ID;
- handle/escalate failures.

Only then acknowledge the Delivery:

```text
orca orchestration check --ack <delivery_id> --json
```

An unacknowledged Delivery is replayed and stalls the loop. A timeout/count zero
is a checkpoint, not a worker failure. Replace a failed worker with a retry of the
same task. If a worker becomes TUI-idle without reporting, nudge it with the exact
completion command from its preamble before using manual task completion.

## Evaluation and decisions

Dispatch one evaluator after all work units satisfy their type pipeline's
readiness rule. Parse the exact verdict block:

- PASS: create/resolve the gate as pass and finish.
- FAIL: create/resolve the gate as fail, increment the implementation attempt,
  and retry with feedback until `max_iterations`.
- HOLD: experiment only; create no gate and consume no iteration. Follow the
  experiment monitoring loop, then dispatch a fresh evaluator.

Account for every settled worker before waiting again. On completion, report the
verdict and cited evidence; at the cap, report unresolved feedback.

## Generic safety

- Do not pass transcripts when a plan/report/ID is sufficient.
- Do not allow workers to weaken tests or alter MLflow evidence.
- Use `new-child` only for a demonstrated shared-write conflict and account for
  missing gitignored files/shared stores in the type pipeline.
