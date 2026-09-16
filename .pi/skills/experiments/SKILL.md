---
name: experiments
description: Reference index for designing, launching, monitoring, and analyzing Gomoku training experiments.
---

# Gomoku experiment references

Read `AGENTS.md` for project architecture and environment conventions. This skill
contains experiment-domain methodology used by Orca workers; role responsibilities
and orchestration live under `.orca/`.

Load only the references needed for the task. Paths are relative to this skill
directory:

- `references/experiment-design.md` — hypotheses, baselines, controls, expected
  metric movement, and evidence conditions.
- `references/training-knobs.md` — curriculum, opponent pool, masks, shaping,
  block reward, defensive openings, snapshots, and evaluation.
- `references/launching-runs.md` — root-level script copies, verification,
  durable background launch, logs, PID, run IDs, and worktree caveats.
- `references/readiness.md` — completion, convergence, plateau requirements,
  dead/stuck runs, and HOLD readiness semantics.
- `references/run-analysis.md` — MLflow layout and per-(heuristic, side) rate plus
  episode-length trajectory comparisons.

The `.orca/roles/` definitions specify which references each worker must read.
The `.orca/pipelines/experiment.md` definition specifies when monitoring and
evaluation occur.
