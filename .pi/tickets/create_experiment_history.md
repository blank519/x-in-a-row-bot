# Create experiment history

type: code
max_iterations: 2

## Goal
Scan over the entire mlruns directory, analyze the results of all past experiments,
and populate the memories directory with experiment history and analysis. This 
includes both individual experiment memories and a global trends memory which 
summarizes the overall research trajectory.

## Constraints
- Do not launch experiments/training runs, modify evidence in `mlruns/`, or make code
  changes. Only create and edit memory files.
- Do not create new tests for this ticket. Instead, evaluate the implementation by
  verifying the created memories against the corresponding evidence in `mlruns/`.
- Try to group runs into experiments that each test a single ticket/hypothesis 
  based on either datetime or modified parameters.

## Done when
- Experiment memory files cover all runs in the mlruns directory.
- A `_TRENDS.json` file exists containing all the required fields.
- All entries must satisfy the schema enforced by the experiment memory tool.