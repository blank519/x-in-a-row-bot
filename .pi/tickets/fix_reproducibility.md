# Fix experiment reproducibility issue

type: code
max_iterations: 3

## Goal
A recent attempt to manually reproduce an agent's run with the same seed and same
exact code showed different results. This issue needs to be fixed to ensure 
reproducibility of experiment results.

## Requirements
- Locate the cause of the reproducibility issue.
- Resolve the issue for all forseeable future runs.

## Done when
- New experiments with arbitrary parameters can be reproduced with the same results.
- If possible, previous experiments (specifically the current best result) can be 
  reproduced. This is not a requirement for a PASS, but it would be ideal.