# Create a tool to monitor ongoing runs

type: code
max_iterations: 3

## Goal
Currently, in order to monitor an ongoing run's progress, the coordinator must write its own commands using `bash`,
which is prone to failure. Create a tool that the coordinator can call to monitor ongoing runs for number of timesteps,
run health, and number of checkpoints, up until the run's completion.

## Constraints
- Create a new script in `.pi/extensions/<tool_name>.ts`.
- The script can monitor `logs/<run_name>.log` for the latest timestep and/or `mlruns/` for the number of checkpoints
  each run has so far (in the end, both measure the same thing).
- The script can monitor PIDs for run health.
- The script should be able to monitor multiple runs at once based on their run names or run IDs.
- The script should sleep for a specified interval between checks.
- The script should be able to stop monitoring when the run is complete.

## Done when
- The tool successfully passes a test which creates two short mock runs and verifies that the tool can monitor their progress until run completion.