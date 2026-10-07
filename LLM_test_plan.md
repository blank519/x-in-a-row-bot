## Goal

Raise Gomoku warmup sample efficiency before any opponent-pool training. Run four independent, one-run probes at 1,024,000 timesteps to identify a configuration whose combined-heuristic X/O and aggregate win-rate trajectories exceed the immutable champion baseline at aligned checkpoints, while tracking offensive and defensive behavior on both sides. A probe can satisfy the ticket's first condition without reaching deployment readiness; readiness remains a stricter, separately reported 60%-on-both-sides decision.

## Type

Experiment (`max_iterations: 1`). Planning only: this artifact does not edit a production/training file, launch training, or create/update experiment memory.

## Acceptance criteria

### Immutable baseline and local evidence

All four runs compare to finished MLflow run **`7bbe675725f941eea81ab0fb38675ade`**, `ppo-gomoku-warmup-block15-15m-2026-08-18`, in experiment `ppo-gomoku` (`mlruns/510583218657647424/7bbe675725f941eea81ab0fb38675ade`). It is the configuration-relevant champion: its terminal `eval/average_win_rate=0.47`, `Combined/x=0.43`, `Combined/o=0.18`, and `worst_win_rate=0.18` exceed the balanced warmup alternatives. Its terminal per-pair rates and lengths are:

| heuristic | learner side | win rate | avg episode length |
|---|---:|---:|---:|
| Combined | X | 0.43 | 11.12 |
| Combined | O | 0.18 | 9.51 |
| Defensive | X | 0.63 | 15.48 |
| Defensive | O | 0.54 | 17.86 |
| Offensive | X | 0.69 | 6.86 |
| Offensive | O | 0.35 | 7.23 |

The strongest isolated local results do not form one balanced model: historical leaders include `Combined/x=0.60` (`c89367b4eaee4bfcbc5dd2de79e0ec00`) but `Combined/o=0.00`, Offensive/O up to 0.51 (`788dfa9797e344f2b993e436698647f3`) while that run finished at only `Combined/x=o=0.03`, Defensive/X up to 0.80 (`89aeea1bc5134df18b0dc4d1b700e387`), Defensive/O up to 0.71 (`63e6289ec9f847c4ad97e070a05994fd`), and Offensive/X up to 0.99 (`f7caf4ff76bf4109940c86e7aaff0bea`). This side imbalance is why aggregate or one-side records alone are not acceptance evidence. Prior higher block reward, larger local masks/global masking, reward shaping, and mixed-heuristic curricula also failed to beat the balanced champion, so none is repeated here.

At the aligned four baseline evaluations (steps 256k, 512k, 768k, 1,024k), `eval/average_win_rate` is `[0.005, 0.015, 0.0067, 0.020]`; Combined X and O remain 0.00 throughout. At 1,024k, the six rates are Combined X/O `0.00/0.00`, Defensive X/O `0.07/0.04`, and Offensive X/O `0.01/0.00`; paired lengths are `6.27/4.48`, `19.19/16.84`, and `5.21/4.00`, respectively. These values make the shortened tests aligned trajectory probes rather than substitutes for full 15.36M-run evidence.

### Observable trajectory success for each 1.024M probe

At completion, compare all four aligned checkpoints, not only the final aggregate. Ticket condition 1 is supported when all three primary trajectories are higher than baseline: the final-two-checkpoint mean of Combined X win rate is at least 0.02, Combined O is at least 0.02, and `eval/average_win_rate` is at least 0.04 (baseline final-two mean is about 0.0133). The final checkpoint must also meet the following six-pair behavioral targets:

| heuristic | X win-rate target | O win-rate target | paired final-length target/interpretation |
|---|---:|---:|---|
| Combined | >=0.03 | >=0.03 | X >=7.00 and O >=5.00, showing longer survival accompanies nonzero wins |
| Defensive | >=0.10 | >=0.08 | X >=17.3 and O >=15.2 (no >10% collapse from baseline while win rate rises) |
| Offensive | >=0.10 | >=0.03 | X >=5.40 and O >=4.20, showing the weak O defense lasts longer rather than preserving four-move losses |

A result below a rate target with longer games is diagnostic only, not success. A result above a rate target with shorter games must be inspected for faster learner wins versus faster losses using the paired loss rate/reward before being credited.

### Exact readiness threshold

The evaluator determines warmup is sufficient for opponent-pool training only if **each** of Combined X and Combined O win rate is >=0.60 at two consecutive 100-game evaluation checkpoints, `eval/average_win_rate>=0.60` at both checkpoints, and none of Defensive/Offensive X/O is below 0.60 at the later checkpoint. This concretizes “consistently win” and preserves the ticket's proposed 60% Combined threshold. Reaching only trajectory success is still success for condition 1 but must be reported as **not yet ready** under condition 2.

## Plan

1. Each implementation starts only from the repository-root `train_ppo_gomoku.py`; it must not read, copy, diff, or otherwise inspect another implementer's script/report. Run `python -m pytest tests -q` from the WSL virtualenv before launch.
2. Every run uses the common controlled configuration below, matching baseline parameters except for the mandatory shortened horizon and its one stated research delta: seed 42; `n_envs=16`; `snapshot_freq=256_000`; feature dimension 512; `reward_shaping_coef=0.0`; `reward_shaping_gamma=0.995`; `block_reward_coef=0.15`; `defensive_opening_prob=0.3`; `n_steps=512`; `batch_size=512`; `start_learning_rate=3e-4`; `final_learning_rate=1e-4`; `gamma=0.995`; `gae_lambda=0.95`; `ent_coef=0.005`; `clip_range=0.1`; `total_timesteps=warmup_steps=mask_learner_until_steps=mask_opponent_until_steps=1_024_000`; `warmup_p_random=0.1`; `warmup_p_heuristics=[0.9]`; `start_mistake_rate=0.7`; `final_mistake_rate=0.1`; `p_random=0.1`; `p_heuristics=[0.4]` (unused before warmup ends); `local_mask_radius=2`; `eval_games_per_side=100`; and `k=50`. Log every listed value and the unit-specific delta to MLflow.
3. The current original has `block_reward_coef=0.4` and a 10.24M horizon. Those are not implicit defaults: each implementer must explicitly set the common `0.15` block coefficient and all four 1.024M schedule/mask values above. No opponent-pool transition may occur.
4. The current learning-rate lambda is `start_learning_rate + p*(final_learning_rate-start_learning_rate)`. SB3's `progress_remaining` (`p`) moves 1 to 0, so this actually rises from 1e-4 to 3e-4. One unit corrects that direction; the other three deliberately retain it as the baseline control schedule so the causal delta remains isolated.
5. Redirect stdout/stderr to the unit/model-specific log, retain the root launch script for reproducibility, and record PID, launch time, immutable candidate run ID, exact MLflow path, baseline ID, and first recheck time in the implementer report. Expected duration is roughly 1–2 hours on the historical hardware; first recheck after 20 minutes, then every 20 minutes until finished/failed. Do not update experiment memories.
6. Evidence is ready only after the MLflow run has a terminal status, four evaluation checkpoints through step 1,024,000, all six win/loss/length/reward series, `eval/average_win_rate`, logged parameters proving the delta, the complete log, and final/best/latest model paths. Compare candidate and baseline at identical steps and report both ticket conditions separately.

### Seven-implementer downstream comparison and collision rule

The downstream comparison will assign **seven independent Pi implementers across these four units**; therefore more than one implementer may independently instantiate the same scientific unit. Each assignment supplies a unique filesystem-safe `model_slug` (lowercase letters, digits, hyphens only, e.g. `claude-sonnet-4-5`); if not supplied, derive it from `$PI_MODEL` by lowercasing and replacing every non-alphanumeric run with `-`. Substitute that slug in every `{model_slug}` token below. This makes every script, run, log, snapshot directory, and model destination model-specific and collision-free while preserving one run per implementation assignment. Implementers are forbidden from reading/copying one another, and all required configuration, targets, paths, and evidence rules are contained in this plan.

## Work units

- id: lr-decay-fix
  independent: yes
  inputs/dependencies: Original `train_ppo_gomoku.py` only; immutable baseline `7bbe675725f941eea81ab0fb38675ade`; no ordering dependency. Apply the common configuration, then change only the LR lambda to `final_learning_rate + p * (start_learning_rate - final_learning_rate)`, producing the intended 3e-4 -> 1e-4 decay. Hypothesis: correcting the reversed schedule gives strong early updates and then stabilizes them, raising all six rates, especially Combined/O and Offensive/O, without shortening their games.
  exact run identity: `run_name="ppo-gomoku-warmup1m-lr-decay-fix-attempt3-{model_slug}-2026-10-03"`; root script `train_ppo_gomoku_warmup1m_lr_decay_fix_attempt3_{model_slug}.py`.
  target trajectory: The shared six-pair targets apply; specifically expect Combined X/O final-two means >=0.03/0.02 and average >=0.05, with Combined final lengths >=7.0/5.0 and Offensive O length >=4.2. Contradiction: primary rates remain at baseline zeros or average final-two mean <0.02 despite the corrected schedule.
  files read: `train_ppo_gomoku.py`; baseline MLflow metric/param files only during evaluation.
  files created or modified: Create only the root script above and `artifacts/raise_warmup_performance_test_attempt3/implement_lr-decay-fix_<attempt>-{model_slug}.md`; do not modify production/training sources.
  runtime artifacts: `logs/ppo-gomoku-warmup1m-lr-decay-fix-attempt3-{model_slug}-2026-10-03.log`; `self_play_snapshots/ppo-gomoku-warmup1m-lr-decay-fix-attempt3-{model_slug}-2026-10-03/`; `outputs/best_vs_heuristic_warmup1m_lr_decay_fix_attempt3_{model_slug}.zip`; `outputs/latest_vs_heuristic_warmup1m_lr_decay_fix_attempt3_{model_slug}.zip`; `outputs/ppo_gomoku_warmup1m_lr_decay_fix_attempt3_{model_slug}.zip`; generated MLflow run tree.
  evidence-ready condition: Terminal run plus four aligned evaluations and every artifact/series required by Plan step 6.
  risks: Correcting LR direction may increase early policy loss or destabilize the very short run; GPU nondeterminism means a marginal threshold crossing is weak evidence.

- id: lr-constant-3e4
  independent: yes
  inputs/dependencies: Original `train_ppo_gomoku.py` only; immutable baseline `7bbe675725f941eea81ab0fb38675ade`; no ordering dependency. Apply the common configuration, then set `final_learning_rate=3e-4` (the existing lambda becomes constant 3e-4); this is the only research-variable change. Hypothesis: a sustained high LR extracts more learning from 1.024M samples than the historical rising 1e-4 -> 3e-4 schedule.
  exact run identity: `run_name="ppo-gomoku-warmup1m-lr-constant3e4-attempt3-{model_slug}-2026-10-03"`; root script `train_ppo_gomoku_warmup1m_lr_constant3e4_attempt3_{model_slug}.py`.
  target trajectory: Shared targets apply; specifically expect final-two means Combined X/O >=0.03/0.02 and average >=0.05, with no Defensive X/O final-rate regression below 0.07/0.04 and no >10% paired-length collapse. Contradiction: volatile checkpoint reversals leave final average <0.02 or either Combined side at zero.
  files read: `train_ppo_gomoku.py`; baseline MLflow metric/param files only during evaluation.
  files created or modified: Create only the root script above and `artifacts/raise_warmup_performance_test_attempt3/implement_lr-constant-3e4_<attempt>-{model_slug}.md`; do not modify production/training sources.
  runtime artifacts: `logs/ppo-gomoku-warmup1m-lr-constant3e4-attempt3-{model_slug}-2026-10-03.log`; `self_play_snapshots/ppo-gomoku-warmup1m-lr-constant3e4-attempt3-{model_slug}-2026-10-03/`; `outputs/best_vs_heuristic_warmup1m_lr_constant3e4_attempt3_{model_slug}.zip`; `outputs/latest_vs_heuristic_warmup1m_lr_constant3e4_attempt3_{model_slug}.zip`; `outputs/ppo_gomoku_warmup1m_lr_constant3e4_attempt3_{model_slug}.zip`; generated MLflow run tree.
  evidence-ready condition: Terminal run plus four aligned evaluations and every artifact/series required by Plan step 6.
  risks: Constant 3e-4 may be too aggressive for PPO and can trade O-side survival for fast X-side offense; evaluator must not accept aggregate-only gains.

- id: entropy-001
  independent: yes
  inputs/dependencies: Original `train_ppo_gomoku.py` only; immutable baseline `7bbe675725f941eea81ab0fb38675ade`; no ordering dependency. Apply the common configuration and retain the historical rising LR lambda, then change only `ent_coef` from 0.005 to 0.001. Hypothesis: less entropy pressure lets the learner exploit the combined heuristic's guided tactical signal sooner, converting longer Combined/Offensive games into early wins on both sides.
  exact run identity: `run_name="ppo-gomoku-warmup1m-entropy001-attempt3-{model_slug}-2026-10-03"`; root script `train_ppo_gomoku_warmup1m_entropy001_attempt3_{model_slug}.py`.
  target trajectory: Shared targets apply; prioritize Combined X/O final >=0.03/0.03 and Offensive X/O >=0.10/0.03, with final lengths Combined >=7.0/5.0 and Offensive >=5.4/4.2. Contradiction: X improves while O remains at zero, indicating premature policy collapse rather than balanced exploitation.
  files read: `train_ppo_gomoku.py`; baseline MLflow metric/param files only during evaluation.
  files created or modified: Create only the root script above and `artifacts/raise_warmup_performance_test_attempt3/implement_entropy-001_<attempt>-{model_slug}.md`; do not modify production/training sources.
  runtime artifacts: `logs/ppo-gomoku-warmup1m-entropy001-attempt3-{model_slug}-2026-10-03.log`; `self_play_snapshots/ppo-gomoku-warmup1m-entropy001-attempt3-{model_slug}-2026-10-03/`; `outputs/best_vs_heuristic_warmup1m_entropy001_attempt3_{model_slug}.zip`; `outputs/latest_vs_heuristic_warmup1m_entropy001_attempt3_{model_slug}.zip`; `outputs/ppo_gomoku_warmup1m_entropy001_attempt3_{model_slug}.zip`; generated MLflow run tree.
  evidence-ready condition: Terminal run plus four aligned evaluations and every artifact/series required by Plan step 6.
  risks: Reduced exploration may overfit first-player tactics and worsen the existing side asymmetry; four checkpoints cannot prove long-run convergence.

- id: batch-256
  independent: yes
  inputs/dependencies: Original `train_ppo_gomoku.py` only; immutable baseline `7bbe675725f941eea81ab0fb38675ade`; no ordering dependency. Apply the common configuration and retain the historical rising LR lambda, then change only `batch_size` from 512 to 256. Hypothesis: twice as many minibatches per PPO epoch improve optimization/sample efficiency enough to lift early defensive and offensive competence on both sides.
  exact run identity: `run_name="ppo-gomoku-warmup1m-batch256-attempt3-{model_slug}-2026-10-03"`; root script `train_ppo_gomoku_warmup1m_batch256_attempt3_{model_slug}.py`.
  target trajectory: Shared targets apply; expect all six final rates to reach the shared minima, average final-two mean >=0.04, and Combined/Offensive O lengths >=5.0/4.2. Contradiction: higher update variance causes any paired length to fall >10% while its loss rate rises, or primary trajectories fail to exceed baseline.
  files read: `train_ppo_gomoku.py`; baseline MLflow metric/param files only during evaluation.
  files created or modified: Create only the root script above and `artifacts/raise_warmup_performance_test_attempt3/implement_batch-256_<attempt>-{model_slug}.md`; do not modify production/training sources.
  runtime artifacts: `logs/ppo-gomoku-warmup1m-batch256-attempt3-{model_slug}-2026-10-03.log`; `self_play_snapshots/ppo-gomoku-warmup1m-batch256-attempt3-{model_slug}-2026-10-03/`; `outputs/best_vs_heuristic_warmup1m_batch256_attempt3_{model_slug}.zip`; `outputs/latest_vs_heuristic_warmup1m_batch256_attempt3_{model_slug}.zip`; `outputs/ppo_gomoku_warmup1m_batch256_attempt3_{model_slug}.zip`; generated MLflow run tree.
  evidence-ready condition: Terminal run plus four aligned evaluations and every artifact/series required by Plan step 6.
  risks: Extra optimizer steps increase wall time and overfitting risk; throughput differences must not be mistaken for timestep-aligned learning gains.

## Risks and assumptions

- The 1.024M cap is a screening horizon. No candidate that merely passes the early trajectory targets is automatically ready for opponent-pool training; the exact 60% readiness rule remains binding.
- Baseline and candidates share seed 42 but CUDA training is not fully deterministic. Treat small (<0.02) one-checkpoint gaps as noise and use the final-two means plus paired behavior.
- One baseline run ID is used for all units to avoid moving-target comparisons. Its 1.024M samples provide exact alignment even though it continued to 15.36M.
- The baseline's high Defensive episode lengths coexist with poor early win rates; length alone can mean indecision. Win/loss/reward must determine whether length movement is beneficial.
- All four units reset the current original's block coefficient from 0.4 to the champion's 0.15. This common control is necessary but means comparisons to a hypothetical unrun 0.4/1.024M source configuration are out of scope.
- Seven independent implementations create duplicated scientific units but not shared writes because `{model_slug}` is mandatory in every writable/runtime path. If two agents resolve to the same slug, orchestration must assign distinct suffixes before launch; agents must not inspect each other's files to resolve it.
- Shared GPU contention can distort duration, not timestep-aligned metrics. Every implementer records PID and log and launches exactly one run.
- No work unit may modify `train_ppo_gomoku.py`, shared environment/policy files, tests, memories, or another unit's artifacts.

## Done when

Exactly four independent one-run configurations have been implemented from the original trainer, pytest has passed for each script, and each MLflow run is evidence-ready at 1,024,000 warmup-only timesteps. Evaluation reports aligned Combined X/O and aggregate trajectories against run `7bbe675725f941eea81ab0fb38675ade`, all six rate/episode-length pairs, and separately states (a) whether ticket condition 1 improved and (b) whether the exact readiness threshold permits opponent-pool training. No experiment memory is created or updated.

## Agent tools used

- `read`: read the complete ticket, planner/common-worker roles, ticket/artifact contracts, `AGENTS.md`, required experiment-design/training-knobs/run-analysis references, orchestration skill, and the original `train_ppo_gomoku.py`.
- `find_mlruns_runs`: ranked local `ppo-gomoku` runs by aggregate and every heuristic/side metric; inspected immutable run IDs, statuses, parameters, aligned histories, terminal rates, and episode lengths.
- `query_experiment_memories`: read-only review of existing canonical trends and relevant experiment conclusions/runs to avoid retesting documented dead ends; no memory write/update tool was used.
- `bash`: loaded the version-matched Orca guide, listed memory names, inspected/summed the Pi session-cost record, validated the artifact, and checked the coordinator inbox; it did not edit production files or launch training.
- `multi_tool_use.parallel`: grouped independent read-only file, memory, MLflow, and inbox inspections to reduce latency.
- `write`: created this planning artifact only.
- `edit`: made targeted corrections to this planning artifact (evidence citation, collision-safe report paths, and the required cost line).

LLM cost through report completion: $1.607039
