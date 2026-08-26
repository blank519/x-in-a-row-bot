# Create a tool to show the latest MLflow run

type: code
max_iterations: 3

## Goal
Currently, the agent must manually read the `mlruns/` directory to see the latest run. Create a tool that locates the latest MLflow run in the directory and displays all of its parameters and metrics, which can be used instead.

## Constraints
- Create a new file `show_latest_run_tool.py` in the root directory.
- Locate runs through the `mlruns/` directory structure, rather than using the MLflow API.
- Display the run ID, parameters, and metrics in a readable format.
- Account for disparity in metric and parameter names across different runs due to updates.

## Done when
- The tool successfully locates and displays the latest MLflow run's information, which is currently named "ppo-gomoku-block-reward-20p-longer-run-more-defensive-opening-2026-08-16-23:47".