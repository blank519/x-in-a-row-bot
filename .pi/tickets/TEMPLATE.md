# <Ticket title>

<!-- One ticket is the pipeline source of truth. Delete comments when filling it. -->

type: code            # code | experiment
max_iterations: 3     # FAIL retry cap; HOLD does not consume an iteration

## Goal
What should be true after this ticket is done, and why it matters.

## Constraints
Anything that must or must not happen: files, APIs, compatibility, runtime,
resource budget, or experiment controls.

## Done when
Explicit, independently observable pass criteria. Describe behavior/metrics, not
an implementation that merely appears likely to produce them.

- For code: required behavior, important edge cases, and compatibility checks.
  The evaluator independently authors acceptance tests.
- For experiments: target metric movement, relevant heuristic/side pairings,
  episode-length interpretation, baseline, and minimum evidence/readiness.

<!-- Experiment tickets only: -->
## Hypothesis
Expected change and causal mechanism.

## Baseline
Immutable MLflow run ID plus descriptive run name, or description of a baseline run
which workers can search using available tools.

## Evidence-ready condition
Completion or the specific sustained plateau/window that permits evaluation.

## Recheck interval
Suggested monitoring interval and expected maximum runtime.
