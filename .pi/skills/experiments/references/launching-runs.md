# Launching and handing off runs

## Before launch

- Work from the repository root in the WSL virtualenv.
- Use the assigned root-level copy of `train_ppo_gomoku.py`, never the shared
  original and never a copy under `artifacts/`; flat imports require repo-root
  placement.
- Apply only the planned delta and a unique run name. Confirm new knobs are logged.
- Run `python -m pytest tests -q` and the code-change smoke checks.

## Durable background launch

Create `logs/`, redirect output, and capture the PID so releasing an Orca terminal
does not lose progress or output:

```bash
mkdir -p logs
nohup python <assigned_script.py> > logs/<run_name>.log 2>&1 & echo $!
```

Wait only until the log contains the training banner and real PPO iteration
progress. Discover the MLflow run ID once its run directory/tag appears. Do not
wait for completion and do not evaluate the run.

## Required handoff

Report:

- run name and MLflow run ID/path;
- assigned script copy and exact configuration delta;
- baseline run ID;
- PID, log path, and launch timestamp;
- evidence of initial progress;
- expected duration and suggested first recheck time.

## Placement and concurrency

Parallel runs may share the active worktree only when every run owns a distinct
script copy and output names do not collide. The shared `mlruns/` store lets one
evaluator see all runs. A new child worktree lacks gitignored `.venv/`, `mlruns/`,
and artifacts by default; if isolation is unavoidable, configure a shared
absolute MLflow tracking URI. GPU VRAM, not CPU availability, is the hard
parallelism limit.
