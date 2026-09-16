# Orca ticket pipeline

Orca is the only orchestration backend in this structure. Planner, implementer,
and evaluator authority is defined entirely under `.orca/`; project-domain
methodology remains in the discoverable `.pi/skills/` references.

## Layout

```text
.orca/
├── coordinator.md
├── contracts/
│   ├── tickets.md
│   └── artifacts.md
├── pipelines/
│   ├── common.md
│   ├── code.md
│   └── experiment.md
└── roles/
    ├── common-worker.md
    ├── planner.md
    ├── implementer.md
    └── evaluator.md
```

- `roles/` defines worker responsibilities, type-specific duties, modification
  boundaries, outputs, and Orca completion behavior.
- `contracts/` defines ticket fields, artifact schemas, and verdict semantics.
- `pipelines/` defines Orca scheduling, monitoring, gates, retries, concurrency,
  and worktree behavior.
- `.pi/skills/code-changes` and `.pi/skills/experiments` provide repository and RL
  methodology without defining agent roles.

## Semantics

- PASS ends the Run.
- FAIL creates a failed gate and retries implementation within
  `max_iterations`.
- HOLD is experiment-only: keep monitoring and reevaluate without consuming an
  iteration or relaunching work.
- Code evaluators may add permanent tests but never production code.
- Experiment evaluators never modify MLflow evidence.

## Worktrees and runs

Use the active worktree by default because tickets, `.venv/`, `mlruns/`, and
artifacts may be uncommitted or gitignored. Parallel experiment units use unique
root-level training script copies and one shared MLflow store. Parallel code
units must have disjoint writes. Use child worktrees only when isolation is
necessary and explicitly configure shared stores/dependencies.

## Running

Start a coordinator session with `.orca/coordinator.md`, then request a ticket:

```text
run the pipeline on .pi/tickets/EXAMPLE-experiment.md
```

The coordinator follows the installed `orchestration` skill to resolve the CLI
and load version-matched command guidance.
