# Create a tool to estimate time until a run completes

type: code
max_iterations: 3

## Goal
Give an accurate estimate of how long a run will take until completion, to 
inform the coordinator about how long to wait for runs to finish.

## Constraints
- Create a new script in `.pi/extensions/<tool_name>.ts`.
- The tool finds a run by its name or ID.
- The tool estimates a run's remaining time until completion from the number 
  of timesteps and time between timesteps, both of which are available in any of 
  the run's metrics.
- If a run has no metrics yet, the tool should wait until metrics are available,
  or a fixed amount of time has passed (currently 30 minutes).

## Done when
- The tool can verifiably estimate the remaining time until completion for test runs.
- The tool demonstrates the ability to wait until metrics are available.
- The tool will return an unknown if metrics are not available after the waiting period.