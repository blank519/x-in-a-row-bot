# Orca experiment-ticket pipeline

Read `.orca/pipelines/common.md`, the experiment skill, and especially its
`launching-runs.md` and `readiness.md` references.

1. Run one planner in the active worktree. Require one work unit per run, a unique
   root-level script copy/run name, baseline ID, target trajectories, and an
   evidence-ready condition.
2. Dispatch independent runs with separate implementers in concurrent waves. Prefer 
   the active worktree: distinct script copies avoid edit conflicts and all runs 
   share `mlruns/`.
3. After every implementer verifies configuration and completes its hand-off, the 
   coordinator launches all runs sequentially and durably, then writes its report.
   a. If there is insufficient VRAM to continue launching runs, the coordinator will
      defer remaining runs and wait until the current runs complete (see step 4), 
      then launch the remaining runs and wait for them to finish as well. This does 
      NOT consume an implementation iteration.
4. The coordinator is then responsible for monitoring runs. Do NOT poll at a 
   short cadence. Instead:
   a. Use the `estimate_run_completion` tool to estimate the completion time for 
      each run while giving time for logs to show up.
   b. Do one early health sweep shortly after launch (process alive, log shows the
      training banner and real PPO progress, MLflow run created) to catch launch
      failures like a dead process or immediate traceback.
   c. Then wait in a SINGLE long blocking interval sized to the plan's expected
      batch-completion time (slowest run/wave plus a grace period).
   d. On waking, sweep every run once for process/log health and target MLflow
      trajectories. The default is to wait until ALL launched runs are
      evidence-ready before evaluating.
   e. Break the wait early ONLY for an errored run: if any run's process has died
      or its log shows an error/traceback, treat that run as failed immediately
      (step 8) and handle/escalate it while the healthy runs keep training — do
      not keep waiting on the dead one.
   f. If runs are healthy but still immature, call `estimate_run_completion` again
      and wait roughly the returned remaining time plus a small grace period, with 
      a floor (e.g. 20 min) to avoid busy-waiting and re-sweep.
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

## Model Configuration

Use the following models for each role:

- **Planner**: 
  - Model Name: GPT 5.6 Sol 
  - Model ID: openai/gpt-5.6-sol
- **Implementer**: 
  - Model Name: GLM 5.3 Flash
  - Model ID: openrouter/z-ai/glm-5.3-flash
- **Evaluator**: 
  - Model Name: GPT 5.6 Sol
  - Model ID: openai/gpt-5.6-sol