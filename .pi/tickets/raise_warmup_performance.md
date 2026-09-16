# Create proficient warmup model before beginning opponent pool training

type: experiment
max_iterations: 5

## Goal
The goal is to raise the RL model's warmup performance to an acceptable level before 
beginning training within an opponent pool.
Analyze the best performing runs on the offensive, defensive, and combined heuristics
on both sides (as X and as O). Determine exact metrics for win rate against each heuristic
that would indicate acceptable performance. Until the model reaches this threshold, identify 
weaknesses and potential improvements based on win rate, episode length, and parameter values. 
Propose a plan to improve the win rate for both offensive and defensive heuristics, and 
execute the plan in `train_ppo_gomoku.py`.

## Constraints
- For each run, create a copy of the current `train_ppo_gomoku.py` file with a descriptive name.
- You must set `total_timesteps` equal to `warmup_steps` (no opponent pool training yet).
- Change any of the other hyperparameters you deem necessary across all runs you start. If
  code changes are required (ex: a lambda to anneal a hyperparameter), make those changes.
- Give each run a descriptive `run_name` (e.g. `ppo-gomoku-block-reward-sweep-...`).
- Verify the code with `python -m pytest tests -q` before launching.

## Hypothesis
Until the model can consistently win against the combined heuristic as both X and O, it will not 
be ready for opponent pool training. My estimate is that a 70% win rate against the combined heuristic
as both X and O is a good threshold.

## Baseline
The best performing run with the highest win rate against the combined heuristic, which should 
also be the run with the highest average win rate.

## Done when
Relative to the baseline, over training:
- `eval/GomokuCombinedHeuristicPolicy/o_win_rate` and `eval/GomokuCombinedHeuristicPolicy/x_win_rate` 
  trend higher, AND
- `eval/average_win_rate` trends higher.
The evaluator determines that the model's win rate against the Combined Heuristic Policy is sufficient
to proceed to opponent pool training.