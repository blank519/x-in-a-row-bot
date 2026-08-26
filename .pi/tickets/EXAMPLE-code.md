# Log a draw rate alongside win/loss in HeuristicEvaluator

type: code
max_iterations: 3

## Goal
`HeuristicEvaluator` currently logs per-(heuristic, side) win and loss rates but
not draw rate. Add a draw rate so experiment analysis can distinguish "learned to
hold a draw" from "still losing".

## Constraints
- Only touch `vs_heuristic_eval.py` (and tests). Do not change the model-selection
  key's behavior.
- Follow existing metric-naming conventions (`{name}/{side}_<suffix>`).

## Done when
- A `{name}/{side}_draw_rate` value is computed and returned in the metrics dict
  for every (heuristic, side) pairing.
- `python -m pytest tests -q` passes, and a new/updated test asserts that
  win_rate + loss_rate + draw_rate == 1.0 for a pairing.
