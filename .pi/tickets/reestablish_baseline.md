# Create proficient warmup model before beginning opponent pool training

type: experiment
max_iterations: 3

## Goal
A bug was discovered that affected the reproducibility of previous runs. While this does not
invalidate existing results, it means that previously established "best results" cannot be
confirmed and cannot be treated as authoritatively best results. Reestablish a baseline for
future experiments to use.

## Requirements
- `max_parallel_workers` = 5
- Look at the best recorded runs in memory and run training with multiple seeds to verify whether
  similar results can be achieved.
- Find the best performing run with similar results to its recorded metrics and use it as the new
  "best recorded run". 
- For the Evaluator: update `TRENDS.json` with the new best recorded run, and flag all prior findings,
  correlations, patterns, etc. in their descriptions with a note that they were established during a 
  previous reproducibility issue.
- Optional: archive some older, well-established findings/runs into historical summaries.

## Hypothesis
The best recorded run is no longer reproducible, meaning that a new verifiable best recorded run must
be established.

## Baseline
N/A

## Done when
- A new best recorded run is established and verified to be reproducible
- `TRENDS.json` is updated with the new best recorded run
- All prior findings, correlations, and patterns are flagged with a note 
  that they were established during a previous reproducibility issue
