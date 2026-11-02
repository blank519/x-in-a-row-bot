# Create a tool to deterministically extract run evidence

type: code
max_iterations: 3

## Goal
Multiple experimental run evaluators wrote their own scripts to extract run 
evidence for analysis. Write a tool for evaluators to use that standardizes the
format and saves time.

## Specifications
- Create a new script in `.pi/extensions/<tool_name>.ts`.
- Locate the evidence extraction scripts created by the evaluators in 
  `artifacts/raise_warmup_performance_<1, 2, 3, and 4>/` to understand what is
  required for this tool in terms of inputs and outputs, and standardize the format.

## Done when
- The tool can extract evidence from MLflow runs in a standardized format.
- The tool can extract evidence from multiple runs at once.
- The tool produces a deterministic, standardized output that can be used for analysis.