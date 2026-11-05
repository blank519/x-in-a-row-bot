## Planner artifact

Write `artifacts/<ticket_name>/plan.md`:

```markdown
## Goal
## Type
## Acceptance criteria
## Plan
1. Atomic step 1
2. Atomic step 2 
3. ...
## Work units
- id: <stable-id>
  independent: yes|no
  inputs/dependencies: ...
  files read: ...
  files created or modified: ...
  assigned steps: ...
  runtime artifacts: ...
## Risks and assumptions
## Done when
```

For code units, enumerate exact writable repository paths. For experiment units,
name the root-level script copy, log path pattern, run name, and expected model /
snapshot destinations. Two units are independent only when all writable paths
and runtime names are disjoint.