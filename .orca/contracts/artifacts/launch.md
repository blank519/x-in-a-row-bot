## Launch report artifact

Write `artifacts/<ticket_name>/launch_attempt_<attempt>.md` after launching
runs:

```markdown
## Launch statuses per run
- <launched, deferred_insufficient_vram, or failed per run>
- <if failed, the reason why>
## Run name and MLflow run ID per run
## WSL PID/process group per run
## Log path and PID-record path per run 
## Launch time per run
## Script path per run
## Model/snapshot destination per run
## Duplicate kill evidence
## Suggested recheck time
- <estimated with `estimate_run_completion` tool>
```