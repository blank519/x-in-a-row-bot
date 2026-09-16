# Gomoku training mechanisms and knobs

Most orchestration lives in `SelfPlaySnapshotCallback` in
`self_play_gomoku.py`; knobs are set in `train_ppo_gomoku.py::main()`, and reward
terms live in `x_in_a_row_sb3_env.py`.

- **Warmup and opponent pool:** before `warmup_steps`, use
  `warmup_p_random`/`warmup_p_heuristics`; afterward use `p_random`,
  `p_heuristics`, and the remaining probability for snapshots.
- **Mistake annealing:** warmup heuristic `mistake_rate` moves from
  `start_mistake_rate` to `final_mistake_rate`; logged as
  `train/mistake_rate`.
- **Snapshots:** every `snapshot_freq`, freeze the current policy into a pool
  capped by `k`. Persistent finetuning can reload a pool.
- **Local masks:** `local_mask_radius`, `mask_learner_until_steps`, and
  `mask_opponent_until_steps` restrict legal consideration near stones early in
  training. Preserve legal-action masking.
- **Potential shaping:** `reward_shaping_coef` and `reward_shaping_gamma` apply
  `gamma*Phi(s') - Phi(s)` where potential is own minus opponent threat mass.
  Match shaping gamma to PPO gamma for policy invariance.
- **Immediate block reward:** `block_reward_coef` rewards reducing opponent
  threat mass and intentionally changes the optimum; logged as
  `train/mean_block_reward`.
- **Defensive openings:** `defensive_opening_prob` and
  `defensive_opening_neighbor_radius` install block-or-lose positions on reset.
- **Evaluation:** `HeuristicEvaluator` evaluates combined, defensive, and
  offensive policies at snapshot boundaries. Best/latest checkpoints are saved.

Any new knob must be logged through `mlflow.log_params(...)`. Each mechanism can
normally be disabled with a zero coefficient/probability, but verify the actual
call path before assuming isolation.
