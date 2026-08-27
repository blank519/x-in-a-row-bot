---
name: experiments
description: Reference for experiment-type work on the Gomoku RL project — running training runs, the training-loop knobs (reward shaping, block reward, curriculum, opponent pool), and analyzing/comparing results in mlruns/. Load this when launching runs or diagnosing results.
---

# Experiments: running runs & analyzing `mlruns/`

On-demand companion to `AGENTS.md` for experiment work. Read `AGENTS.md` for
project orientation (setup, architecture, conventions); read this for how to run
and analyze experiments.

## Training loop components

The Gomoku training loop layers several mechanisms on top of plain self-play.
Most are orchestrated by `SelfPlaySnapshotCallback` (`self_play_gomoku.py`) with
knobs set in `train_ppo_gomoku.py::main()`; the reward terms live in the env
(`x_in_a_row_sb3_env.py`). Each can be disabled independently (usually by setting
its coefficient/probability to 0).

- **Warmup vs. opponent pool (curriculum stages).** While
  `num_timesteps < warmup_steps` the opponent is a fixed mixture of random moves
  and the combined heuristic (`warmup_p_random`, `warmup_p_heuristics`). After
  warmup the main `OpponentPoolPolicy` is used: random with prob `p_random`,
  heuristics with probs `p_heuristics`, and the remaining mass drawn from the
  self-play snapshot pool.
- **Heuristic mistake-rate annealing.** During warmup the combined heuristic's
  `mistake_rate` is linearly annealed `start_mistake_rate -> final_mistake_rate`,
  so the opponent starts beatable (positive learning signal for blocking) and
  ramps to full strength. Logged as `train/mistake_rate`.
- **Self-play snapshotting.** Every `snapshot_freq` steps the current model is
  frozen and added to the opponent pool (capped at `k`). The finetune scripts can
  also reload a persisted snapshot pool from disk.
- **Local move masking.** `apply_local_move_mask` restricts both the learner and
  the opponent to cells within a Chebyshev `local_mask_radius` of existing stones
  (or near board center when empty), until `mask_learner_until_steps` /
  `mask_opponent_until_steps`. Keeps early play local/tactical.
- **Potential-based reward shaping** (`reward_shaping_coef`, `reward_shaping_gamma`).
  Dense, *policy-invariant* signal (Ng et al. 1999): potential `Phi` = own threat
  mass minus opponent threat mass summed over `win_con`-length windows; shaped
  reward is `gamma*Phi(s') - Phi(s)`. Set `reward_shaping_gamma` == PPO `gamma`.
- **Immediate block reward** (`block_reward_coef`). Rewards the learner for
  reducing the opponent's threat mass with its own move. Unlike potential shaping
  this *does* change the optimal policy (deliberately, to incentivize blocking).
  Exposed per-step as `info["block_reward"]`; logged as `train/mean_block_reward`.
- **Defensive-opening curriculum** (`defensive_opening_prob`,
  `defensive_opening_neighbor_radius`). With the given probability, `reset()`
  installs a designed "block or lose" position (opponent has 3–4 in a row, learner
  to move) instead of an empty board, teaching defense that sparse self-play never
  reaches. See `SingleAgentSelfPlayEnv._install_defensive_opening`.
- **Evaluation & best-model saving.** On each snapshot boundary
  `HeuristicEvaluator` (`vs_heuristic_eval.py`) plays the model against the
  combined/defensive/offensive heuristics; the best and latest checkpoints are
  saved and `eval/*` metrics logged.

## Running experiments

Launching a training run **is** the core research activity here — do it freely.

- **Launch a Gomoku run:** `python train_ppo_gomoku.py`. All hyperparameters,
  reward coefficients, curriculum settings, and the `run_name` are hardcoded in
  `main()`; to test a hypothesis, edit those values, then run. There are no CLI
  flags for training config.
- **Continue an existing run:** `python finetune_ppo_persistent_pool.py`. This
  loads an existing model and continues training from where it left off, with the
  same opponent pool if that has been enabled. You can change the hardcoded model
  path and hyperparameters if it fits the experiment.
- **Name runs after the hypothesis.** Follow the existing convention
  (`run_name=f"ppo-gomoku-<short-description>-{date}"`) so runs are self-describing
  in `mlruns/`. Log any new knob you introduce via `mlflow.log_params(...)` so it
  shows up alongside the metrics when comparing.
- **Runs are long** (the default is ~10M timesteps and can take hours on GPU).
  Start them in the **background** and keep working: monitor progress via stdout
  and the `eval/*` metrics that land in `mlruns/` on every snapshot boundary
  (`snapshot_freq`). The best/latest checkpoints are saved to `outputs/` as they
  improve, so a run is useful even before it finishes.
- **Fast iteration.** To sanity-check a mechanism quickly, temporarily shrink
  `total_timesteps` / `warmup_steps` (and optionally the board) — but treat those
  as smoke tests, not as evidence: only compare full-length runs against each
  other, since curriculum stages and eval are step-dependent.
- **Reproducibility.** `seed` is set (default 42), but GPU kernels are not
  fully deterministic; expect small run-to-run variation and prefer trends /
  multiple runs over single-run differences.
- **Verify code first.** Before a long run, confirm the change is sound with
  `python -m pytest tests -q` and the smoke checks (see the `code-changes` skill)
  — don't burn GPU hours on a run that crashes at the first eval.

## Analyzing & comparing runs

**Do not diagnose a run from its final aggregate numbers.** A single
`average_win_rate` (or even `worst_loss_rate`) at the end of training hides the
dynamics that actually explain *why* a model plays the way it does. The
aggregate leaderboard is only useful as a coarse index of which runs exist.

The analysis that has produced real insight on this project is **per-(heuristic,
side) trajectories over timesteps** of two quantities together:

1. **Win/loss rate vs. each individual heuristic, split by side (X vs. O).**
   A model which struggles against a specific heuristic may reveal specific
   weaknesses.
2. **Average evaluation episode length, per (heuristic, side), over timesteps.**
   Episode length is a strong behavioral proxy: very short games mean decisive
   quick outcomes, long games mean the model is contesting/defending. Reading it
   alongside win rate tells you *how* the model is winning or losing, not just
   whether it is.

Watch how these **change over timesteps** within a run: rising win rate with
lengthening games against an aggressive opponent indicates the model is learning
to defend and contest; a win rate that climbs then collapses, or episode lengths
that shrink over time, indicates the model is converging onto — or diverging away
from — a particular tactic. Compare the *shapes* of these curves across runs, not
just their endpoints.

**Worked example (why these two signals matter).** A prior run's model struggled
specifically against the **offensive** heuristic when playing **second (as O)**,
*and* its average episode length in those games was **< 5 moves**. Together those
two facts showed the model was not defending at all — it was losing almost
immediately rather than contesting. That diagnosis (invisible in the aggregate
win rate) directly motivated adding the **defensive-opening curriculum** and the
**block reward**. Reproduce that kind of reasoning: pair a per-opponent/side rate
with its episode-length trajectory before concluding anything.

**Finding runs to compare** (by name, date, params — no UI needed):

Recall the section on reading '/mlruns' directly (`AGENTS.md`).

- Resolve the experiment id by grepping the experiment `meta.yaml` files for the
  name (`grep -r "name: ppo-gomoku" mlruns/*/meta.yaml`). `ppo-gomoku` is
  currently `510583218657647424`, but treat ids as opaque.
- List runs and their names by reading each `<run_id>/tags/mlflow.runName` (run
  names encode the hypothesis and date by convention). `start_time`/`end_time` in
  `meta.yaml` are epoch-ms if you need exact timing.
- Read `params/<name>` files to filter/compare configurations. To diff two runs,
  compare their `params/` directories (e.g. what changed between a baseline and a
  candidate).

**Reading a metric trajectory over timesteps.** Each metric file has one line per
logged point, whitespace-separated:

```
<timestamp_ms> <value> <step>
```

`<step>` is the training timestep, and lines are already in step order — so
reading a file top-to-bottom gives you the full curve. For the per-(heuristic,
side) analysis above, the files are, for each heuristic
`H` in {`GomokuCombinedHeuristicPolicy`, `GomokuDefensiveHeuristicPolicy`,
`GomokuOffensiveHeuristicPolicy`} and side `s` in {`x`, `o`}:

- `metrics/eval/H/s_win_rate`, `metrics/eval/H/s_loss_rate` — per-opponent/side rates.
- `metrics/eval/H/s_avg_episode_length` — the episode-length diagnostic.
- `metrics/eval/H/s_avg_reward_per_game` — average return.
- Aggregates (coarse only): `metrics/eval/average_win_rate`,
  `metrics/eval/average_loss_rate`, `metrics/eval/worst_win_rate`,
  `metrics/eval/worst_loss_rate`, `metrics/eval/improved`.
- Training diagnostics: `metrics/train/mistake_rate`,
  `metrics/train/mean_block_reward`.

For example, reading `metrics/eval/GomokuOffensiveHeuristicPolicy/o_avg_episode_length`
alongside `.../o_win_rate` for one run shows the value and step on each line, so
you can see directly whether games are lengthening as the O-side win rate rises
(defending) or staying short (the "not defending" pattern from the worked
example). Read a few metric files across two runs and compare the columns.

Supporting notes:

- **What the trainer calls "best"** (from `HeuristicEvaluator`,
  `vs_heuristic_eval.py`): the best-model selection key prioritizes, in order,
  lower `average_loss_rate`, higher `average_win_rate`, lower `worst_loss_rate`,
  higher `worst_win_rate` (worst-case across every heuristic/side pairing). Useful
  to know, but it is a selection rule, not a substitute for the per-opponent
  trajectory analysis above.
- **Older runs may lack newer params/metrics** — a missing `params/<name>` file or
  `metrics/.../<key>` file tells you which mechanism a run predates. Compare
  against the most relevant baseline, not just the newest run.
- The `mlflow` Python API (`mlflow.search_runs`,
  `MlflowClient().get_metric_history`) also works headless and returns the same
  data as objects if you prefer querying in Python — but reading the files
  directly is usually the simplest path for the analysis above.
