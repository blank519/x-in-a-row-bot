# Does a higher block reward improve O-side defense vs the offensive heuristic?

type: experiment
max_iterations: 2

## Goal
Test whether increasing `block_reward_coef` helps the model defend when playing
second (as O) against the offensive heuristic, without hurting overall win rate.

## Constraints
- Change only `block_reward_coef` relative to the baseline; keep all other
  hyperparameters, curriculum settings, and total_timesteps identical.
- Give the run a descriptive `run_name` (e.g. `ppo-gomoku-block-reward-sweep-...`).
- Verify the code with `python -m pytest tests -q` before launching.

## Hypothesis
A larger immediate block reward gives a stronger, earlier signal to interrupt the
opponent's threats, so the model should contest O-side games instead of losing
them quickly.

## Baseline
The most recent completed run with the current `block_reward_coef`
(find it in `mlruns/` under experiment `ppo-gomoku`; compare against it).

## Done when
Relative to the baseline, over training:
- `eval/GomokuOffensiveHeuristicPolicy/o_win_rate` trends higher, AND
- `eval/GomokuOffensiveHeuristicPolicy/o_avg_episode_length` increases (games are
  contested, not lost in < ~5 moves), AND
- `eval/average_win_rate` does not regress meaningfully.
