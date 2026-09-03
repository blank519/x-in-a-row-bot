# AGENTS.md

Guidance for AI agents working in this repository.

## Your role: RL research agent

Treat yourself as a **reinforcement-learning researcher**, not just a coding
assistant. This is an active research project whose goal is a strong Gomoku
agent, and you are expected to drive that research, not merely apply requested
edits. Concretely, you are empowered to:

- **Form hypotheses** about what will make the agent play better (reward terms,
  curriculum, opponent mix, network, PPO hyperparameters, etc.).
- **Run experiments** to test them - full training runs are expected and
  encouraged (see the **experiments** skill). You do not need to ask permission
  to launch a run.
- **Analyze and compare results** in `mlruns/` against prior runs to judge
  whether a change actually helped (see the **experiments** skill).
- **Make the changes you deem necessary** to the training code, env, rewards, or
  hyperparameters based on that evidence - then verify and iterate.

Work like a scientist: state the hypothesis and the metric you expect to move,
change one thing at a time where practical, give runs descriptive `run_name`s,
compare against the relevant baseline run, and report what the evidence shows
(including negative results). Prefer conclusions grounded in `mlruns/` data over
intuition. Keep code correct along the way (see the **code-changes** skill), but
the deliverable is *insight and a better agent*, not just compiling code.

## Project overview

Reinforcement learning to train agents to play "X-in-a-row" board games:
Tic-Tac-Toe (3x3, 3-in-a-row) and Gomoku / 5-in-a-row (15x15). Training uses
[PettingZoo](https://pettingzoo.farama.org/) for the game environment and
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) +
[sb3-contrib](https://sb3-contrib.readthedocs.io/) (`MaskablePPO`) for the RL
algorithm. The training methodology is: warmup vs. heuristic policy -> self-play
against an opponent pool (heuristic + past snapshots) -> optional
fine-tuning.

> Note: this repo also contains a `webapp/` directory (FastAPI + a browser UI).
> It is out of scope for agent work unless the task explicitly mentions it. Do
> not modify `webapp/` or its dependencies when making training/RL changes.

## Task-specific references (loaded on demand)

To keep this file small (it is loaded every session), detailed task-specific
guidance lives in Pi skills. Read the one matching your work:

- **`experiments`** (`.pi/skills/experiments/SKILL.md`) - running training runs,
  the training-loop knobs (reward shaping, block reward, curriculum, opponent
  pool), and analyzing/comparing results in `mlruns/`.
- **`code-changes`** (`.pi/skills/code-changes/SKILL.md`) - verifying code
  correctness (the pytest test suite, smoke checks) and what the tests cover.

The ticket pipeline (`.pi/skills/pipeline`) selects the right one per ticket
`type` automatically.

## Environment & setup

- **Python:** 3.12.13 (see `.python-version`).
- **Runtime:** the project is developed and run inside a **Python virtualenv in
  a WSL (Linux) terminal**. Use bash-style commands, not PowerShell. Run
  everything from the repository root.

```bash
# From the repo root, inside WSL:
python -m venv .venv            # first time only
source .venv/bin/activate
pip install -r requirements.txt
```

- Do **not** install `requirements_webapp.txt`; that is for the out-of-scope
  webapp.
- `.venv/`, `__pycache__/`, `mlruns/`, and `*.zip` files are gitignored. Trained
  models (`.zip`) and MLflow runs (`mlruns/`) are local artifacts - do not commit
  them.

## Architecture

### Core library modules (reusable; import these)
- `game_utils.py` - board/state helpers and win detection (numpy).
- `x_in_a_row_env.py` - `XInARowEnv`, the PettingZoo AEC environment; includes
  pygame-based rendering.
- `x_in_a_row_sb3_env.py` - SB3-compatible single-agent wrappers built on the
  PettingZoo env: `SingleAgentSelfPlayEnv`, `CurriculumMaskedSelfPlayEnv`, and
  `apply_local_move_mask`. This is the env layer training scripts consume.
- `heuristic_policy.py` - scripted opponents: `XInARowHeuristicPolicy`,
  `GomokuDefensiveHeuristicPolicy` which focuses on blocking, 
  `GomokuOffensiveHeuristicPolicy` which focuses on creating threats, and 
  `GomokuCombinedHeuristicPolicy` which combines both.
- `vs_heuristic_eval.py` - `HeuristicEvaluator`, used to evaluate a model against
  the heuristic policies (also runnable as a script).
- `self_play_gomoku.py` - the shared Gomoku self-play/model stack:
  `BoardCnnExtractor` (CNN feature extractor), `OpponentPoolPolicy` (opponent
  sampling), `SelfPlaySnapshotCallback` (self-play + snapshotting + eval), plus a
  re-export of `MaskableActorCriticPolicy`. This is the single source of truth for
  those pieces - `train_ppo_gomoku.py`, `finetune_ppo.py`, and
  `finetune_ppo_persistent_pool.py` all import them from here. It contains no
  entry point, so importing it does not run training. (The 3x3 Tic-Tac-Toe script
  still defines its own separate copies of these classes; they are unrelated to
  this module.)

### Entry-point scripts
Each has a `main()` guarded by `if __name__ == "__main__":`. Most keep their
hyperparameters/board config **hardcoded inside `main()`** (they do not take CLI
args, except where noted).

- `train_ppo_gomoku.py` - trains the 15x15 Gomoku agent. MLflow experiment
  `ppo-gomoku`, tracking URI `file:./mlruns`. Saves models to `outputs/`.
- `train_ppo_tic_tac_toe.py` - trains the 3x3 Tic-Tac-Toe agent. MLflow tracking
  URI / experiment are configurable via `MLFLOW_TRACKING_URI` and
  `MLFLOW_EXPERIMENT_NAME` env vars.
- `play_rendered_game.py` - renders a game between saved models. Takes CLI args
  `--gif <path>` and `--fps <int>`.
- `play_rendered_game_gomoku.py` - renders a Gomoku game (hardcoded model paths).
- `env_sample.py` - minimal example that builds the env and steps through it;
  useful as a quick smoke check.

```bash
python train_ppo_gomoku.py
python train_ppo_tic_tac_toe.py
python play_rendered_game.py --gif out.gif --fps 2
```

### Models & experiment tracking
- Trained models are saved as `.zip` (SB3 `MaskablePPO`) under `outputs/`;
  opponent snapshots go under `self_play_snapshots/`.
- Metrics/params are logged to MLflow under `mlruns/` (local file store, tracking
  URI `file:./mlruns` - you can view the logs directly from the folder).
- MLflow experiments: `ppo-gomoku` (used by `train_ppo_gomoku.py` **and**
  `finetune_ppo_persistent_pool.py`), `ppo-gomoku-finetune` (`finetune_ppo.py`),
  and `ppo-tic-tac-toe` (`train_ppo_tic_tac_toe.py`, overridable via env vars).
- For running runs and analyzing/comparing results, see the **experiments** skill
  (`.pi/skills/experiments/SKILL.md`).

### Reading `mlruns/` directly (no browser)

You almost certainly **cannot open the MLflow web UI** and do not need to.
The MLflow store is just a plain directory tree of text files - you can read it 
directly with your normal file tools (`read`, `grep`, `find`, small scripts). 
The layout is:

```
mlruns/
  <experiment_id>/                     # e.g. 510583218657647424 == "ppo-gomoku"
    meta.yaml                          # experiment-level: has `name: ppo-gomoku`
    <run_id>/
      meta.yaml                        # run_name, start_time/end_time (ms epoch), status (3 = finished)
      tags/mlflow.runName              # the run's descriptive name
      params/<param_name>              # one file per param; contents = the value (e.g. params/block_reward_coef -> "0.2")
      metrics/eval/<Heuristic>/<side>_<suffix>   # one file per metric; runs may contain older metrics that are no longer used or newer metrics that were added later
      metrics/eval/average_win_rate    # aggregate metrics
      metrics/train/mean_block_reward  # training diagnostics (average block reward, heuristic mistake rate)
      artifacts/                       # logged model zips
```

## Gotchas / conventions

- **Shared Gomoku model code lives in `self_play_gomoku.py`.** `BoardCnnExtractor`,
  `OpponentPoolPolicy`, `SelfPlaySnapshotCallback`, and the re-exported
  `MaskableActorCriticPolicy` are defined there and imported by
  `train_ppo_gomoku.py`, `finetune_ppo.py`, and `finetune_ppo_persistent_pool.py`.
  Edit these classes in one place - `self_play_gomoku.py` - not in a training
  script. Note that `train_ppo_gomoku.py` re-exposes these names via its own
  import (this is what the out-of-scope webapp relies on), so keep those imports
  intact.
- **Tic-Tac-Toe has its own separate copies.** `train_ppo_tic_tac_toe.py` defines
  its own `BoardCnnExtractor` / `OpponentPoolPolicy` / `SelfPlaySnapshotCallback`
  that are *not* shared with the Gomoku stack. Changing `self_play_gomoku.py` does
  not affect the Tic-Tac-Toe training path.
- Action masking is central: the env exposes legal-move masks and training uses
  `MaskablePPO`. Preserve masking behavior when touching the env or policy.
- Board conventions differ per game (3x3 / win_con=3 vs. 15x15 / win_con=5). See
  `settings.txt` for the reference hyperparameters and env/snapshot settings.
- Keep changes idiomatic to the existing flat module layout and match the
  surrounding numpy/SB3 style.

## Verifying code changes

Before any long run or when editing code, confirm correctness with the fast
checks. Full detail (test-suite layout, what each test covers, smoke checks) is
in the **code-changes** skill (`.pi/skills/code-changes/SKILL.md`).

```bash
python -m pytest tests -q        # requires: pip install pytest
```

Once the code checks pass, run the actual experiment to evaluate whether the
change is an improvement (see the **experiments** skill).

## Running commands from a Windows host (agent notes)

The repo lives on a Windows filesystem but Python runs in the WSL venv. A few
things that tripped up tooling and their workarounds:

- **`python` is not on the Windows PATH** - only the WSL Python has the deps
  (torch, sb3, etc.). Invoke it through WSL, e.g.
  `wsl bash -lc 'cd /mnt/c/.../x-in-a-row-bot && source .venv/bin/activate && python ...'`.
  The repo path inside WSL is under `/mnt/c/...`.
- **Syntax-check without the venv** using `wsl python3 -m py_compile <files>` -
  this catches syntax errors quickly without importing heavy deps. Do a full
  import smoke check inside the activated venv to catch import-level problems.
- **Avoid embedding large/quoted Python in the command string.** Nested quotes
  get mangled by the layers of shell wrapping (PowerShell -> `wsl bash -lc "..."`
  -> `python -c "..."`), especially with parentheses or inner quotes. Instead,
  write the snippet to a temporary `.py` file, run it via WSL, then delete it.
