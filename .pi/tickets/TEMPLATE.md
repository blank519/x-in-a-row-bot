# <Ticket title>

<!--
A ticket is the single input to the pipeline (see .pi/skills/pipeline). The
planner reads it, the implementer executes, and the evaluator judges against
"Done when". Fill in the fields below; delete these comments.
-->

type: code            # or: experiment
max_iterations: 3     # optional; orchestrator retry cap (default 3)

## Goal
What should be true after this ticket is done, and why it matters.

## Constraints
Anything that must or must not happen (files to avoid, style, budget, etc.).

## Done when
The explicit, checkable pass criteria the evaluator will verify. Be concrete.
- For `code`: which tests must pass / behavior must hold.
- For `experiment`: which metric must move, in which direction, vs. which baseline.

<!-- Experiment tickets should also fill in: -->
## Hypothesis        (experiment only)
What you expect to change and the mechanism behind it.

## Baseline          (experiment only)
The run_name / mlruns run id to compare against.
