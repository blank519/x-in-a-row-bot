# Condense _TRENDS.json into a curated summary

type: code
max_iterations: 3

## Goal
In order to prevent `_TRENDS.json` from growing too large for a context-limited model, 
remove and condense some of the information it contains, and update the tool 
`experiment_memories.ts` to handle the new format.

## Constraints
- Change the evidence field to only store run_id or experiment_name pointers plus their
  timestamps.
- Remove the qualification field and move its reasoning into a conclusion within 
  per-experiment memory files, where it already belongs.
- Turn older findings and dead ends into a one-line summary without evidence 
  (e.g., "heuristic-heavy curricula, 8 runs, Jul --> dead end").

## Done when
- The experiment memory tool schema now creates an evidence field that only stores
  run_id or experiment_name pointers plus their timestamps. 
- All existing evidence in `_TRENDS.json` is converted to the new format.
- The experiment memory tool no longer creates qualification fields in `_TRENDS.json`, and
  instead creates a "conclusion" or "summary" field in the per-experiment memory files.
- The qualification field is removed and its reasoning is moved to conclusion/summary fields 
  in the per-experiment memory files.
- The experiment memory tool is able to condense older findings and dead ends into one-line 
  summaries without evidence.
- Older findings and dead ends are condensed into one-line summaries without evidence.