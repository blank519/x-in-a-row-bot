# Orca experiment-ticket pipeline

Read `.orca/pipelines/common.md`, the experiment skill, and especially its
`launching-runs.md` and `readiness.md` references.

1. Run one planner in the active worktree. Require one work unit per run, a unique
   root-level script copy/run name, baseline ID, target trajectories, and an
   evidence-ready condition.
2. Dispatch independent runs in concurrent waves. Prefer the active worktree:
   distinct script copies avoid edit conflicts and all runs share `mlruns/`.
3. Each implementer verifies configuration, launches durably, confirms real PPO
   progress, and reports run ID/PID/log/baseline/recheck time. Worker completion
   means launched, not trained.
4. After releasing launch workers, the coordinator owns monitoring. Check process
   and log health plus target MLflow trajectories for every run at the interval
   defined by `readiness.md` or the plan.
5. Dispatch one batch evaluator only when every run is complete or converged. It
   compares each candidate with its baseline using per-(heuristic, side) rate and
   episode-length trajectories.
6. PASS/FAIL follows the common gate/retry loop. HOLD creates no gate, launches no
   replacement run, and consumes no iteration: monitor and reevaluate.
7. A dead run with no usable evidence is FAIL. A live but stuck run is bounded by
   expected duration plus a grace period, then escalated.

Use a child worktree only when scripts or outputs genuinely conflict. Since child
worktrees omit gitignored `.venv/`, `mlruns/`, and artifacts, configure a shared
absolute MLflow tracking URI before using one. Never evaluate separate invisible
stores as if they were one batch.
