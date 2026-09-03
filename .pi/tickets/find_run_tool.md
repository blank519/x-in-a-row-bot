# Create a tool to locate MLflow runs by time, metrics, or parameters

type: code
max_iterations: 3

## Goal
Currently, the agent must manually read the `mlruns/` directory to locate any run. Create a tool for Pi agents to use which locates any MLflow runs in the directory based on time (e.g., latest run, or specific date range), metrics (e.g., best average win rate, or best win rate against the Offensive Heuristic policy as player O), or parameters (e.g., specific hyperparameters), and displays all of their parameters and metrics, which can be used instead. The tool should be able to return multiple runs based on criteria.

## Constraints
- Create a new script in `.pi/extensions/<tool_name>.ts`.
- Locate runs through the `mlruns/` directory structure, rather than using the MLflow API.
- For every returned run, display the run ID, parameters, and metrics in a readable format.
- Account for disparity in metric and parameter names across different runs due to updates.

## Done when
- The tool successfully passes tests to locate and display the latest three MLflow runs' information, and locate the three runs with the best average win rate.