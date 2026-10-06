# Launching and handing off runs

## Coordinator launch

After the implementation units settle, the coordinator should launch the prepared
runs with the `launch_run` tool, providing:
- `scriptPath`;
- `runName`;
- optional `requiredFreeVramMiB` (omit unless plan specifies a non-default requirement);
- optional `startupTimeoutSeconds`. 

Do not construct `nohup`, WSL, redirection, duplicate-kill, or PID-discovery commands manually.

A successful tool call is evidence of launch, not experiment completion.

## Reporting Launch

For every batch of launches, successful or failed, create a launch report at 
`artifacts/<ticket_name>/launch_attempt_<attempt>.md`. Read the launch report template in 
`.orca/contracts/artifacts.md` for the required fields.

On failure, record the tool's stated reason and cleanup evidence. Do not
substitute a manual launch.

## Placement and concurrency

Parallel runs may share the active worktree only when every run owns a distinct
script copy and output names do not collide. The shared `mlruns/` store lets one
evaluator see all runs. A new child worktree lacks gitignored `.venv/`, `mlruns/`,
and artifacts by default; if isolation is unavoidable, configure a shared
absolute MLflow tracking URI. GPU VRAM, not CPU availability, is the hard
parallelism limit.
