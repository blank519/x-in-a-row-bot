# Project-local Pi resources used by the Orca pipeline

This project runs its orchestration pipeline through Orca. The defunct Pi
subagent role and pipeline definitions are not part of the proposed structure.
The `.pi/` directory remains the home of project tickets and domain skills that
Orca workers read as Markdown references.

## Layout

```text
.pi/
├── skills/
│   ├── code-changes/       # code correctness and test-suite reference
│   └── experiments/        # experiment design/run/readiness/analysis references
└── tickets/                # shared ticket inputs and templates
```

Role authority, artifact contracts, verdict semantics, and orchestration live
under `.orca/`:

```text
.orca/
├── coordinator.md
├── contracts/
├── pipelines/
└── roles/
```

## Skills

`code-changes` documents repository-specific correctness checks and existing test
coverage. `experiments` is a short index into focused references for experiment
design, training knobs, durable launching, evidence readiness, and MLflow
trajectory analysis.

These files use Pi skill frontmatter for discovery, but Orca workers can also
read them directly by path. They intentionally do not define planner,
implementer, evaluator, or orchestration behavior.

## Tickets

Tickets under `.pi/tickets/` remain the source input to the Orca coordinator.
`Done when` should be independently observable. Experiment tickets should name an
immutable baseline run ID, target per-(heuristic, side) behavior, and an
evidence-ready condition.
