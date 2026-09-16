---
name: code-changes
description: Repository reference for verifying code correctness and understanding existing test coverage.
---

# Code changes: correctness and verification

Read `AGENTS.md` for architecture, environment, and repository invariants. Role
responsibilities and test authorship are defined under `.orca/roles/`; this file
contains the shared code-change methodology and repository-specific commands.

Verification here establishes code correctness. Whether a change improves playing
strength requires an experiment and MLflow comparison.

## Change boundaries

- Shared Gomoku model components belong in `self_play_gomoku.py`, not duplicated
  edits in each Gomoku training entry point.
- Tic-Tac-Toe keeps separate model-stack definitions.
- Preserve legal-action masking and 3x3/15x15 configuration boundaries.
- Ordered changes or changes writing the same files cannot safely run as parallel
  work units.

## Acceptance-test guidance

Tests derived from a ticket should verify externally observable `Done when`
behavior and important edge cases rather than mirror implementation structure.
Use a stable feature/ticket filename so later evaluation attempts extend tests
instead of creating duplicates. Permanent regressions belong under `tests/`;
temporary diagnostic scripts belong under the ticket artifact directory.

A failing acceptance test should normally be resolved by correcting production
behavior. If a test contradicts the ticket, report the exact conflict rather than
weakening it silently.

## Verification commands

Run from the repository root inside the WSL virtualenv:

```bash
python -m pytest tests -q
python -c "import game_utils, x_in_a_row_env, x_in_a_row_sb3_env, heuristic_policy, vs_heuristic_eval, self_play_gomoku"
python env_sample.py
```

Use targeted tests while diagnosing, then run the complete suite before claiming
correctness.

## Existing coverage

- `tests/test_selfplay_env.py`: complete episodes against random, callable
  heuristic, and trained-agent-style opponents for base and curriculum wrappers;
  also locality masking.
- `tests/test_defensive_opening.py`: defensive-opening probability and installed
  puzzle invariants.
- `tests/conftest.py`: places the repository root on `sys.path`.

New behavior not covered by this suite needs a durable acceptance/regression test.
