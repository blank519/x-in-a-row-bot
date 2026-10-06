# Create a tool to launch experimental runs

type: code
max_iterations: 3

## Goal
Currently, an agent must build the launch command from scratch for every run. This is 
fine for powerful LLMs, but less reliable for weaker ones. Write a tool that launches
a created run and confirms its successful launch.

## Constraints
- Create a new script in `.pi/extensions/<tool_name>.ts`.
- The tool should build a launch command for a training run.
- The tool redirects the output to `log` files.
- The run should run in the background so it survives if the terminal is closed.
- The tool should return the process ID of the launched run.
- The tool will perform a VRAM preflight to confirm that there is enough VRAM to launch the run.
- The tool waits to confirm that the run has started successfully by waiting until 
  the log shows run progression and/or metrics are available in the `mlruns/<run_id>` directory.
- If an existing run with the same name already exists, it will be killed.

## Done when
- A low power worker agent (such as Qwen 3.8 27B, GLM 5.3 Flash, or DeepSeek V4 Flash) can 
  successfully launch a training run using the new tool on the very first attempt, without 
  creating any "orphan" or "zombie" runs that never complete.
- The tool returns the process ID of the launched run on success.
- The tool returns failure if the run fails to launch for any other reason, such as lack of VRAM, 
  and cites the reason why.
- The tool has proven ability to kill an existing duplicate run.