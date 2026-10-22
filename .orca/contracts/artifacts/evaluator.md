## Evaluator artifact

Write `artifacts/<ticket_name>/evaluate_<attempt>.md`:

```markdown
## Evidence
## Files created
- <test, diagnostic, plot, table, or none>
## Files modified
- <test path and reason, or none>
## Runtime artifacts inspected
- <run/log/model/metric paths, or none>
## Cleanup / retained evidence
## Assessment
## Verdict
VERDICT: PASS|FAIL|HOLD
FEEDBACK: <blank for PASS; actionable correction or missing evidence otherwise>
```

Evaluator-created permanent tests belong under `tests/`; temporary diagnostics
belong under the ticket artifact directory. The evaluator lists, but never
modifies, experiment runtime artifacts. On retries, update an existing stable
acceptance-test file when appropriate rather than creating duplicate attempt
files.

The `VERDICT:` and `FEEDBACK:` lines must be the final two lines. HOLD is valid
only for experiment tickets.