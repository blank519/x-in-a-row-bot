# Orca code-ticket pipeline

Read `.orca/pipelines/common.md` and `.pi/skills/code-changes/SKILL.md`.

1. Run one planner in the active worktree.
2. Dispatch disjoint, independent code units concurrently up to the configured
   worker limit. Group shared-file or ordered changes into one sequential unit.
3. Use the active worktree for disjoint units. If units actually write the same
   files, do not overlap them; prefer ordering over a child-worktree merge burden.
4. Each implementer runs the existing suite and smoke checks before handoff.
5. After all implementation units finish, dispatch one evaluator over the full
   diff. It independently adds acceptance/regression tests under `tests/`, runs
   the complete suite, and does not modify production code.
6. PASS/FAIL follows the common gate/retry loop. HOLD is invalid for code tickets.

On FAIL, preserve evaluator-authored tests for the next attempt. The next
implementer fixes production behavior and must not weaken those tests. Escalate a
test that demonstrably contradicts the ticket instead of silently changing it.
