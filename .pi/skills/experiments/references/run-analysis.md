# Comparing MLflow runs

Do not judge a run from one final aggregate. Compare candidate and baseline
**trajectories over aligned timesteps** for:

1. win/loss rate against each heuristic, split by learner side X/O; and
2. average episode length for the same heuristic/side pair.

These signals together distinguish quick tactical failure from longer contested
play. For example, low O-side win rate against the offensive heuristic combined
with games under five moves indicates failure to defend, while increasing game
length alongside improving win rate supports learned defense.

## Finding runs

Resolve experiment IDs from `mlruns/*/meta.yaml`, identify runs using
`tags/mlflow.runName`, and verify immutable run IDs. Compare `params/` directories
to ensure the proposed delta is the actual delta. Older runs may lack newer
metrics or parameters; choose a configuration-relevant baseline.

Metric files contain one point per line:

```text
<timestamp_ms> <value> <training_step>
```

For each heuristic in `GomokuCombinedHeuristicPolicy`,
`GomokuDefensiveHeuristicPolicy`, and `GomokuOffensiveHeuristicPolicy`, and each
side `x`/`o`, inspect:

- `metrics/eval/<H>/<side>_win_rate`
- `metrics/eval/<H>/<side>_loss_rate`
- `metrics/eval/<H>/<side>_avg_episode_length`
- `metrics/eval/<H>/<side>_avg_reward_per_game`

Use aggregates only as coarse summaries:

- `metrics/eval/average_win_rate`, `average_loss_rate`
- `metrics/eval/worst_win_rate`, `worst_loss_rate`
- `metrics/eval/improved`

Training diagnostics include `metrics/train/mistake_rate` and
`metrics/train/mean_block_reward`.

## Evaluation standard

Cite run IDs, steps, recent trajectory values/ranges, configuration differences,
and the paired behavioral interpretation. PASS only if the ticket's expected
movement is supported without violating constraints. FAIL if mature evidence
contradicts it. HOLD if a live incomplete run is not ready under `readiness.md`.

The trainer's best-model key prioritizes lower average loss, higher average win,
lower worst loss, then higher worst win. It is a checkpoint-selection rule, not
a substitute for per-opponent trajectory analysis.
