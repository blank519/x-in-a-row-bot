## Implementer artifact

Write `artifacts/<ticket_name>/implement_<unit_id>_<attempt>.md`:

```markdown
## Completed
## Files created
- <path> — <purpose; permanent|runtime|temporary>
## Files modified
- <path> — <change>
## Verification done
## Cleanup / outstanding files
## Notes / handoff
```

List files even when they are gitignored. `git diff` is not a substitute because
it omits logs, MLflow runs, models, snapshots, and other runtime output. Report
unexpected partial files and failed-launch artifacts rather than silently leaving
them behind.

For experiments, the handoff must include run name, exact configuration difference,
and known model/snapshot destinations.