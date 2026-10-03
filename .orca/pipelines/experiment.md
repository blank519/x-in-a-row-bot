# Orca experiment-ticket pipeline

Read `.orca/pipelines/common.md`, the experiment skill, and especially its
`launching-runs.md` and `readiness.md` references.

1. Run one planner in the active worktree. Require one work unit per run, a unique
   root-level script copy/run name, baseline ID, target trajectories, and an
   evidence-ready condition.
2. Dispatch independent runs with separate implementers in concurrent waves. Prefer 
   the active worktree: distinct script copies avoid edit conflicts and all runs 
   share `mlruns/`.
3. Each implementer verifies configuration, launches durably, confirms real PPO
   progress, and reports run ID/PID/log/baseline/recheck time. Worker completion
   means launched, not trained.
4. After releasing launch workers, the coordinator owns monitoring. Do NOT poll at
   a short cadence. Instead:
    a. Do one early health sweep shortly after launch (process alive, log shows the
      training banner and real PPO progress, MLflow run created) to catch launch
      failures like a dead process or immediate traceback.
   b. Then wait in a SINGLE long blocking interval sized to the plan's expected
      batch-completion time (slowest run/wave plus a grace period).
   c. On waking, sweep every run once for process/log health and target MLflow
      trajectories. The default is to wait until ALL launched runs are
      evidence-ready before evaluating.
   d. Break the wait early ONLY for an errored run: if any run's process has died
      or its log shows an error/traceback, treat that run as failed immediately
      (step 8) and handle/escalate it while the healthy runs keep training — do
      not keep waiting on the dead one.
   e. If runs are healthy but still immature, estimate remaining time and wait 
      roughly that plus a small grace period, with a floor (e.g. 10 min) to avoid 
      busy-waiting and re-sweep.
5. Dispatch one batch evaluator only when every run is complete or converged. It
   compares each candidate with its baseline using per-(heuristic, side) rate and
   episode-length trajectories.
6. Evaluator should record experiment memories by appending per-run entries and
   updating trends memory if applicable. The worker should not be released until 
   memory recording is finished.
7. PASS/FAIL follows the common gate/retry loop. HOLD creates no gate, launches no
   replacement run, and consumes no iteration: monitor and reevaluate.
8. A dead run with no usable evidence is FAIL. A live but stuck run is bounded by
   expected duration plus a grace period, then escalated.

Use a child worktree only when scripts or outputs genuinely conflict. Since child
worktrees omit gitignored `.venv/`, `mlruns/`, and artifacts, configure a shared
absolute MLflow tracking URI before using one. Never evaluate separate invisible
stores as if they were one batch.
