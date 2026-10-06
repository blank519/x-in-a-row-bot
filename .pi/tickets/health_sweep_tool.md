# Create a tool to perform health sweep after launching experimental runs

type: code
max_iterations: 3

## Goal
Turn the required health sweep after launching experimental runs into a tool for consistency,
especially when using weaker LLMs. This is to catch launch failures like a dead process or 
immediate traceback early on.

## Requirements
- Create a new script in `.pi/extensions/<tool_name>.ts`.
- The tool should take as input a list of run names, their run IDs, and/or their PIDs.
- The tool should check the health of each run by checking that each process is alive, that 
  the run's log shows the training banner and real PPO progress, and that the MLflow run was 
  created.
- The tool should return a list of unhealthy runs, or an empty list if all runs are healthy.
- The tool should wait a fixed amount of time (e.g. 10 minutes) for the run log to be populated
  with the training banner and real PPO progress.

## Done when
- A low power worker agent (such as Qwen 3.8 27B, GLM 5.3 Flash, or DeepSeek V4 Flash) can 
  successfully use the new tool to perform a health sweep immediately after launching a training 
  run.
- The tool returns a list of unhealthy runs, or an empty list if all runs are healthy.
- The tool will wait for the run log to be populated for a fixed amount of time.