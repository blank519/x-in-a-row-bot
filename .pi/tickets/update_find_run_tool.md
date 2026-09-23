# Update find_mlruns_runs tool to work with full run history

type: code
max_iterations: 3

## Goal
Currently, when `find_mlruns_runs.ts` is used to search for runs based on metrics, it only
checks and returns the value of the latest timestep. Add an option to search the entire history
of timesteps for a given metric, and rank runs based on values at any timestep. Additionally,
add an option to return the entire history of timesteps for returned metrics.

## Details
- Update the file `find_mlruns_runs.ts`.
- Add an option to search the entire history of timesteps for a given metric or a fixed interval
  of recent timesteps (e.g., last 10 timesteps), instead of just the latest timestep.
- When ranking returned runs by a metric, add a flag to specify whether the ranking should be based
  on the latest value, the max/min across the entire history, or some aggregation like recent mean.
- Add an option to return specific searched metrics and parameters for a run, instead of all of them.
  Run name and ID should always be returned.
- For returned metrics, include an option to return the entire history of timesteps or a fixed 
  interval of recent timesteps.

## Done when
- A Pi worker is able to use the tool to locate and display MLflow runs' information correctly.
- The tool works with any search criteria (run name/ID, time, metrics, parameters).
- The tool can return multiple runs based on search criteria.
- The tool can handle missing metrics and parameters across different runs due to updates.
- The tool can provably perform every single feature mentioned above as OPTIONAL features.