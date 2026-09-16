# Designing an experiment

Treat each full training run as a test of a stated causal hypothesis.

## Required plan for each run

- Hypothesis and mechanism.
- Relevant baseline run name and immutable MLflow run ID.
- Exact parameter/code delta, naming variables in
  `train_ppo_gomoku.py::main()` or the environment.
- Distinct `run_name` following `ppo-gomoku-<short-description>-<date>`.
- A distinct root-level script copy such as
  `train_ppo_gomoku_<short_name>.py` so parallel workers never edit one file.
- Target per-(heuristic, side) rates and paired episode-length movement that
  would support or contradict the hypothesis.
- Evidence-ready condition and expected duration/recheck interval.

Change one variable at a time where practical. If several changes are inseparable,
state the confound. Prefer multiple seeds when resources permit: seed 42 is
configured, but GPU training is not fully deterministic.

A shortened timestep/board run can validate mechanics but is not comparative
research evidence. Compare full-length runs at aligned timesteps and curriculum
stages. Select the most configuration-relevant baseline, not merely the newest
run, and inspect parameter differences before planning.
