# Create a tool to locate MLflow runs by any variable

type: code
max_iterations: 3

## Goal
Currently, the agent must manually read the `mlruns/` directory to locate any run. Create a tool for Pi agents to use which locates any MLflow runs in the directory based on any variable - run name/ID, time (e.g., latest run, or specific date range), metrics (e.g., best average win rate, or best win rate against the Offensive Heuristic policy as player O), or parameters (e.g., specific hyperparameters), and displays all of their parameters and metrics, which can be used instead. 

## Constraints
- Create a new script in `.pi/extensions/<tool_name>.ts`.
- The tool should be able to return multiple runs based on search criteria.
- The tool should locate runs through the `mlruns/` directory structure, rather than using the MLflow API.
- The tool should be able to handle missing metrics and parameters across different runs due to updates.

## Done when
- A Pi worker is able to use the tool to locate and display MLflow runs' information correctly.
- The tool works with any search criteria (run name/ID, time, metrics, parameters).
- The tool can return multiple results based on search criteria.
- The tool can handle missing metrics and parameters across different runs due to updates.