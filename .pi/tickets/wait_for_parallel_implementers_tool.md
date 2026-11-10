# Create tool to wait for all implementers to finish

type: code
max_iterations: 3

## Goal
Create a tool that allows the coordinator to wait for all currently running implementers 
to finish running, instead of stopping the command 
`orca orchestration check --wait --types worker_done,escalation,question --timeout-ms 900000 --json` 
every time a single implementer finishes.

## Constraints
- Create a new script in `.pi/extensions/<tool_name>.ts`.
- The tool should wait when implementer workers are live or unverifiable, and return 
  when ALL implementer workers have finished running.
- The tool should NOT acknowledge or hide messages from workers.
- The tool should NOT release or clean up workers.
- The tool should interrupt its waiting period IF AND ONLY IF anything requiring
  coordinator action is detected (e.g., questions, escalations).
- The tool should have a maximum waiting time argument, which is treated as a checkpoint
  rather than a failure.

## Done When
- A coordinator agent can use the tool to monitor multiple dispatched Implementer workers.
- The tool returns a "completed" result once all implementer workers have finished running, 
  plus a list or other small summary of all outcomes.
- The tool returns a "timeout" result if the maximum waiting time is reached.
- The tool immediately returns an "action required" result plus the details of the action 
  required if anything requiring coordinator action is detected.
- The coordinator can validate all completion messages together, then perform the required
  release, acknowledgment, and cleanup in one turn.
