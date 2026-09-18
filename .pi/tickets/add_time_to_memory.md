# Add Timestamps to Memories and Memory Tool

type: code
max_iterations: 3

## Goal
Add timestamps to the experiment memory tool's schema. Then revise all runs in all 
experiment memory files and the trends memory file to include timestamps using the new
schema. This way, the workers understand what findings are outdated and can prioritize
newer directions.

## Constraints
- Do not launch experiments/training runs, modify evidence in `mlruns/`, or create new
  files. Only edit `experiment_memories.ts` and use it to edit existing memories to 
  include timestamps.

## Done when
- The experiment memory tool's schema now enforces timestamps on runs.
- Run entries in experiment memory files all have an associated and correct timestamp.
- All runs referenced in `_TRENDS.json` have an associated and correct timestamp.
