# Orca worker lifecycle adapter

Orca prepends a lifecycle preamble containing `task_id`, `dispatch_id`, and the
exact `orca orchestration send --type worker_done ...` command.

After writing the artifact required by the shared role contract, your final
action must be to run that exact command in your shell. Put a concise summary in
`--body`, list files you changed in `--files-modified`, and report exactly once
with `--outcome succeeded` or `--outcome failed`. Do not print the command as
text, represent it as JSON, or continue working after sending completion.
