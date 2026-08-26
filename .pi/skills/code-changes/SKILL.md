---
name: code-changes
description: Reference for code-type work on the Gomoku RL project — how to verify code correctness (pytest test suite, smoke checks) and what the existing tests cover. Load this when editing code or judging a code change.
---

# Code changes: verifying correctness

On-demand companion to `AGENTS.md` for code work. Read `AGENTS.md` for project
orientation and conventions; read this for how to verify a change.

Verification here is about **code correctness**, not about whether a change helps
the agent play better — that question is answered by running an experiment and
comparing runs (see the `experiments` skill). Use the fast checks below to
confirm a change is sound *before* spending GPU hours on a training run.

**Test suite** (`tests/`, pytest — not in `requirements.txt`, install once with
`pip install pytest`):

```bash
python -m pytest tests -q
```

The suite is fast (no torch, no training) and covers:
- `tests/test_selfplay_env.py` — the self-play env runs a full episode to
  termination against every opponent type (random, heuristic callable, and a
  trained-agent-style `.predict` object), for both the base env and the
  `CurriculumMaskedSelfPlayEnv` used in training; plus a locality-masking check.
- `tests/test_defensive_opening.py` — the `defensive_opening_prob` gate installs
  puzzles at the expected rate, and each installed puzzle has the opponent
  holding 3 or 4 in a row, the learner to move, and all learner stones within the
  neighbour radius of the threat line.

`tests/conftest.py` puts the repo root on `sys.path`, so `pytest` works from the
repo root regardless of import mode.

**Smoke checks** (no pytest needed):

```bash
# Imports resolve and core modules load without error:
python -c "import game_utils, x_in_a_row_env, x_in_a_row_sb3_env, heuristic_policy, vs_heuristic_eval, self_play_gomoku"

# Env builds and steps:
python env_sample.py
```

When you add reusable logic, add a matching fast test under `tests/` so future
changes stay verifiable. Once the code checks pass, run the actual experiment to
evaluate whether the change is an improvement (see the `experiments` skill).
