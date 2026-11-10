# Create execution memory tool

type: code
max_iterations: 5

## Goal
Points of repeated agent difficulty may indicate a need for changes in the
agent's given instructions, a new tool that should be created, or repeatable 
patterns in execution that could be automated. Create a tool to make agents
flag points of difficulty or commonly repeated action sequences and store 
them in memory for review.

## Constraints
- Create a script that writes memory files in the directory `memories/tooling/`.
- Choose a file formatting that is human-readable and machine-parseable.
- Each memory file should correspond to a single flagged difficulty or repeated
  action sequence.
- Memory files should follow a fixed structure.
- Each memory file should include a running tally of how frequently this problem
  is encountered. Accordingly, the tool should be able to search for similar
  problems and update their tallies.

## Done when
- A worker can create memory files that can contain a flagged difficulty or 
  repeated action sequence entry.
- When a worker flags a problem, the tool can search for similar problems and 
  correctly determine whether or not it is the same problem. 
- If a matching problem is found, the tool updates the tally of an existing memory 
  file instead of creating a new memory file. If not, it creates a new file.
- Content may be placeholder text; what matters is schema enforcement and locatability.