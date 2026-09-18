# Create tools to generate and read memories of experiments

type: code
max_iterations: 5

## Goal
So that experimental agents don't restart analysis from scratch each run, build a
memory system they can use to store and retrieve analysis of previous runs: a
detailed per-experiment record plus a curated cross-run summary. This ticket
builds only the tooling; the actual analysis content will be produced later by the
pipeline (placeholder content is fine here).

## Constraints
- Provide a write/edit tool and a read/query tool (two scripts, or one
  script with subcommands — your call).
- Memory files should be stored under `memories/experiments/` under a 
  machine-parseable format for Pi workers to read and edit without full-text grep,
  while remaining human-readable.

### Writing and Editing Memories
- There should be one memory file per experiment named for the experiment/ticket
  (`memories/experiments/<experiment_name>.<ext>`), plus a single canonical
  trends memory file (`memories/experiments/_TRENDS.<ext>`).
- The **experiment memory files** should each correspond to one experiment (= one 
  ticket/hypothesis), with one entry appended per run. Each run entry should contain
  these fields:
  - `run_name`
  - `mlruns_run_id` (and/or path): links the entry to its evidence in `mlruns/`
  - `modified_params`: param name -> value changed vs. baseline
  - `goal_hypothesis`: the hypothesis being tested
  - `results_insights`: should reference the key metric(s) that support it
  - `reasoning`: a list of **at least 3** candidate explanations, ordered by
    likelihood (the tool enforces the ≥3 list; it does not invent the reasons)
- The **trends memory file** (`_TRENDS.<ext>`) should contain the overall research 
  trajectory of all experiments. It contains, as distinct sections:
  - chronological breakthroughs and dead ends
  - correlations and patterns observed across runs
  - the current most promising run + parameters (the champion)
  - the most promising directions to continue research
- Editing one section must preserve the others.

### Reading Memories
- The read/query tool should be able to locate entries by experiment, `run_name`, 
  modified param, metric/outcome, or tag, and return the matching structured run 
  entry(ies), not a whole file dump.
- The read/query tool should be able to read the canonical trends file and locate 
  a requested section quickly.

## Done when
- A worker can create an experiment memory file, append a run entry for an
  arbitrary existing `mlruns` run, and read it back; a second append adds a new
  entry without modifying the first.
- The write tool rejects/flags a run entry missing a required field or with
  fewer than 3 reasons.
- A worker can query/filter the store to locate a specific run entry by
  `run_name` (and by a modified param) without full-text grep.
- A worker can create and edit the single canonical `_TRENDS` file so it holds
  all required sections, with an edit to one section preserving the others, and
  can locate a requested section quickly.
- Content may be placeholder text; what matters is schema enforcement,
  append-without-clobber, and locatability.
