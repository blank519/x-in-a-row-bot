# Unit: LRCONST-S44 (ticket reestablish_baseline, attempt 2, iteration 2 of
# at most 3 per artifacts/reestablish_baseline/plan_2.md).
# Root-level copy of the unit's retained attempt-1 script
# train_ppo_gomoku_ppo-gomoku-rebaseline-ent010-seed44-2026-11-07.py, reconstructed to
# reproduce the scientific configuration of the immutable pre-fix diagnostic
# champion MLflow run b9299ce03cdf46da8b0f9b969396c15f
# ("ppo-gomoku-warmup-wideproj1024-entropy010-block15-15m-2026-10-14",
# experiment ppo-gomoku / 510583218657647424; historical source
# artifacts/raise_warmup_performance_4/train_ppo_gomoku_ppo-gomoku-warmup-wideproj1024-entropy010-block15-15m-2026-10-14.py)
# for the five-seed rebaseline cohort. The champion is a comparison target, not
# authoritative evidence after the previous reproducibility issue.
# Immutable matched same-seed attempt-1 control: MLflow run
# df88a1038ff54408a47b8e8ccfc44704
# ("ppo-gomoku-rebaseline-ent010-seed44-2026-11-07", same experiment).
# Exact delta versus the matched attempt-1 control: ONLY the learning-rate
# schedule. The control's historical rising expression
# lambda p: 3e-4 + p*(1e-4-3e-4) (SB3 progress_remaining p goes 1 -> 0, so the
# rate rises 1e-4 -> 3e-4) is replaced by the scalar constant
# learning_rate=1e-4 (MaskablePPO accepts a scalar and keeps it constant).
# Every other scientific knob is unchanged from attempt 1: 15x15/win-5,
# 16 environments, n_steps=batch_size=512, gamma and shaping gamma 0.995,
# GAE 0.95, entropy 0.010, clip 0.1, no potential shaping, block reward 0.15,
# defensive-opening probability 0.30, Combined-only warmup (0.1 random,
# [0.9] heuristic) with mistakes 0.7 -> 0.1, local radius 2, 100 evaluation
# games per side, snapshot frequency 256,000, and the mandatory four-way
# equality total_timesteps == warmup_steps == mask_learner_until_steps ==
# mask_opponent_until_steps == 15_360_000 (entire run is warmup; no
# opponent-pool training). Seed remains 44, eval_seed 0, and the accepted
# post-fix reproducibility implementation is preserved: strict
# configure_reproducibility(seed) before environment creation or any CUDA
# query, and the deterministic script-local wide-projection extractor whose
# shape probe uses a zero tensor in eval mode (never the observation space's
# sampling API, never mutating BatchNorm statistics).
# Run name is static; do not rename. This script is the unit's reproducible
# launch configuration and must be retained through final evaluation.

import torch as th
import torch.nn as nn

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from sb3_contrib import MaskablePPO

from x_in_a_row_sb3_env import CurriculumMaskedSelfPlayEnv
from self_play_gomoku import MaskableActorCriticPolicy, SelfPlaySnapshotCallback
from reproducibility import configure_reproducibility, log_reproducibility_metadata, policy_state_sha256

import mlflow


class DeterministicWideProjectionBoardCnnExtractor(BaseFeaturesExtractor):
    """Script-local deterministic wide-projection extractor for unit LRCONST-S44.

    The required post-fix counterpart of the immutable champion's wide
    extractor: the same five-convolution stack ([64, 64, 128, 128, 128]
    channels, 3x3 kernels, padding 1, BatchNorm/ReLU order), flattening, and
    widened post-flatten projection Linear(n_flatten, 1024) -> ReLU ->
    Linear(1024, features_dim) -> ReLU. The only difference from the historical
    pre-fix extractor is the shape probe: it uses a zero tensor in eval mode
    (never the observation space's sampling API), so the unseeded space RNG is
    not consumed and BatchNorm running statistics stay at their initial values;
    the prior training flag is restored afterwards. No architectural change.
    """

    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Shape probe on a zero tensor, never the observation space's sampling
        # API: an unseeded space RNG and a training-mode probe would both
        # perturb the BatchNorm running statistics, making same-seed
        # constructions unequal. Eval mode during the no-grad probe keeps every
        # running statistic at its initial value; the prior training flag is
        # restored afterwards.
        was_training = self.cnn.training
        self.cnn.eval()
        try:
            with th.no_grad():
                probe = th.zeros((1, *observation_space.shape), dtype=th.float32)
                n_flatten = self.cnn(probe).shape[1]
        finally:
            self.cnn.train(was_training)

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 1024),  # Champion wide projection (unchanged).
            nn.ReLU(),
            nn.Linear(1024, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations.float()))


def make_env(height: int, width: int, win_con: int,
             reward_shaping_coef: float = 0.0, reward_shaping_gamma: float = 0.99,
             block_reward_coef: float = 0.0,
             defensive_opening_prob: float = 0.0):
    def _thunk():
        return CurriculumMaskedSelfPlayEnv(
            height=height,
            width=width,
            win_con=win_con,
            p1_symbol="X",
            p2_symbol="O",
            render_mode=None,
            opponent_policy="random",
            randomize_learner=True,
            learner_local_mask_radius=None,
            reward_shaping_coef=reward_shaping_coef,
            reward_shaping_gamma=reward_shaping_gamma,
            block_reward_coef=block_reward_coef,
            defensive_opening_prob=defensive_opening_prob,
        )

    return _thunk


def main():
    # Static, collision-checked identity for this unit.
    run_name = "ppo-gomoku-rebaseline-lrconst1e4-ent010-seed44-2026-11-08"
    seed = 44
    eval_seed = 0

    height = 15
    width = 15
    win_con = 5

    # Strict determinism BEFORE environment creation or any other CUDA query.
    configure_reproducibility(seed)

    # Run-specific destinations (disjoint from all sibling units).
    snapshot_dir = "self_play_snapshots/ppo-gomoku-rebaseline-lrconst1e4-ent010-seed44-2026-11-08"
    best_model_path = "outputs/best_vs_heuristic_ppo-gomoku-rebaseline-lrconst1e4-ent010-seed44-2026-11-08"
    latest_model_path = "outputs/latest_vs_heuristic_ppo-gomoku-rebaseline-lrconst1e4-ent010-seed44-2026-11-08"
    final_model_path = "outputs/ppo-gomoku-rebaseline-lrconst1e4-ent010-seed44-2026-11-08"
    snapshot_freq = 256_000 # Close to 250k, multiple of batch_size * n_environments

    # Immutable baseline configuration (b9299ce03cdf46da8b0f9b969396c15f).
    # Potential-based reward shaping disabled, matching the baseline.
    reward_shaping_coef = 0.000
    reward_shaping_gamma = 0.995
    # Immediate block reward, baseline value 0.15.
    block_reward_coef = 0.15
    # Defensive-opening curriculum, baseline value 0.3.
    defensive_opening_prob = 0.3

    # PPO parameters (baseline values).
    n_steps=512
    batch_size=512
    # Attempt-2 mechanism delta (the ONLY scientific change vs. attempt 1):
    # constant learning rate 1e-4 replaces the historical rising expression
    # lambda p: 3e-4 + p*(1e-4-3e-4) (SB3 progress_remaining p: 1 -> 0, so the
    # historical expression actually RISES from 1e-4 to 3e-4). The control
    # schedule value is logged for provenance alongside the constant value.
    learning_rate = 1e-4
    gamma=0.995
    gae_lambda=0.95
    ent_coef=0.010
    clip_range=0.1

    # Training schedule: entire run is warmup, ending exactly at the warmup
    # boundary (baseline values). The four-way equality below is mandatory and
    # prohibits opponent-pool training; the shared callback may create one
    # isolated snapshot exactly at the equality boundary.
    total_timesteps = 15_360_000
    warmup_steps = 15_360_000

    # Opponent pool parameters (baseline values).
    # For warmup: p_random + sum(p_heuristics) + p_snapshot should equal 1.0
    warmup_p_random = 0.1
    warmup_p_heuristics = [0.9] # Combined heuristic
    start_mistake_rate = 0.7 # Initial chance of combined heuristic making a mistake
    final_mistake_rate = 0.1 # Mistake rate at the end of the warmup anneal

    # For self-play: p_random + sum(p_heuristics) < 1.0, remainder = snapshot pool
    # (logged but unused because learning ends at the warmup boundary).
    p_random = 0.1
    p_heuristics = [0.4]
    local_mask_radius = 2
    mask_learner_until_steps = warmup_steps  # mask learner during warmup only
    mask_opponent_until_steps = mask_learner_until_steps + 0  # keep opponent local slightly longer than learner
    eval_games_per_side = 100

    n_envs = 16
    env = DummyVecEnv([
        make_env(
            height, width, win_con,
            reward_shaping_coef=reward_shaping_coef,
            reward_shaping_gamma=reward_shaping_gamma,
            block_reward_coef=block_reward_coef,
            defensive_opening_prob=defensive_opening_prob,
        )
        for _ in range(n_envs)
    ])
    env.seed(seed)
    device = "cuda" if th.cuda.is_available() else "cpu"

    print(
        f"[Train] device={device} "
        f"cuda_available={th.cuda.is_available()} "
        f"torch_version={th.__version__}"
    )
    print(
        f"[Unit] run_name={run_name} "
        f"unit=LRCONST-S44 attempt=2 ticket=reestablish_baseline "
        f"unit_delta=learning_rate_rising_1e-4_to_3e-4_to_constant_1e-4 "
        f"seed={seed} eval_seed={eval_seed} "
        f"feature_extractor=DeterministicWideProjectionBoardCnnExtractor "
        f"projection_hidden_dim=1024 features_dim=512 "
        f"baseline=b9299ce03cdf46da8b0f9b969396c15f"
    )


    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ppo-gomoku")

    with mlflow.start_run(run_name=run_name):
        # Provenance FIRST inside the run: strict backend flags, source
        # manifest/hash, git and runtime versions (seed/eval_seed included).
        log_reproducibility_metadata(
            seed=seed,
            eval_seed=eval_seed,
            device=device,
            entrypoint_path=__file__,
        )

        mlflow.log_params({
            # Identity / provenance
            "run_name": run_name,
            "attempt": "2",
            "unit_id": "LRCONST-S44",
            "ticket": "reestablish_baseline",
            "unit_delta": "learning_rate_rising_1e-4_to_3e-4_to_constant_1e-4",
            "matched_control_run_id": "df88a1038ff54408a47b8e8ccfc44704",
            "matched_control_run_name": "ppo-gomoku-rebaseline-ent010-seed44-2026-11-07",
            "baseline_run_id": "b9299ce03cdf46da8b0f9b969396c15f",
            "baseline_run_name": "ppo-gomoku-warmup-wideproj1024-entropy010-block15-15m-2026-10-14",
            "script_source": "train_ppo_gomoku_ppo-gomoku-rebaseline-ent010-seed44-2026-11-07.py",
            "script_path": "train_ppo_gomoku_ppo-gomoku-rebaseline-lrconst1e4-ent010-seed44-2026-11-08.py",
            "reproducibility_impl": "post_fix_strict_configure_reproducibility_zero_probe_extractor",
            # Architecture (champion reconstruction; unchanged in this unit)
            "feature_extractor": "DeterministicWideProjectionBoardCnnExtractor",
            "projection_hidden_dim": 1024,
            "features_dim": 512,
            "conv_channels": "[64,64,128,128,128]",
            "conv_receptive_field": 11,
            "board": "15x15",
            "win_con": win_con,
            # Environment / PPO
            "n_envs": n_envs,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "learning_rate_schedule": "constant_1e-4",
            "learning_rate": learning_rate,
            "control_learning_rate_schedule": "linear_rising_1e-4_to_3e-4_progress_remaining_baseline_expression",
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "ent_coef": ent_coef,
            "clip_range": clip_range,
            "snapshot_freq": snapshot_freq,
            "warmup_steps": warmup_steps, # guided play before full self-play
            "warmup_p_random": warmup_p_random,
            "warmup_p_heuristics": warmup_p_heuristics,
            "start_mistake_rate": start_mistake_rate,
            "final_mistake_rate": final_mistake_rate,
            "p_random": p_random,
            "p_heuristics": p_heuristics,
            "local_mask_radius": local_mask_radius,
            "mask_learner_until_steps": mask_learner_until_steps, # mask learner during warmup only
            "mask_opponent_until_steps": mask_opponent_until_steps, # keep opponent local slightly longer than learner
            "eval_games_per_side": eval_games_per_side,
            "total_timesteps": total_timesteps, # Entire run will be warmup
            "reward_shaping_coef": reward_shaping_coef,
            "reward_shaping_gamma": reward_shaping_gamma,
            "block_reward_coef": block_reward_coef,
            "defensive_opening_prob": defensive_opening_prob,
            "snapshot_pool_k": 50,
            # Run-specific destinations
            "snapshot_dir": snapshot_dir,
            "best_model_path": best_model_path,
            "latest_model_path": latest_model_path,
            "final_model_path": final_model_path,
        })

        model = MaskablePPO(
            policy=MaskableActorCriticPolicy,
            env=env,
            verbose=1,
            policy_kwargs={
                # Champion reconstruction: wire this script's policy_kwargs to
                # the script-local deterministic wide-projection extractor
                # (1024 hidden, features_dim 512); no architectural delta.
                "features_extractor_class": DeterministicWideProjectionBoardCnnExtractor,
                "features_extractor_kwargs": {"features_dim": 512},
            },
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,  # Constant 1e-4: the single scientific delta.
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            clip_range=clip_range,
            device=device,
            seed=seed,
        )

        self_play_cb = SelfPlaySnapshotCallback(
            vec_env=env,
            snapshot_dir=snapshot_dir,
            snapshot_freq=snapshot_freq,
            height=height,
            width=width,
            win_con=win_con,
            k=50, # max snapshot pool size
            warmup_steps=warmup_steps, # guided play before full self-play
            warmup_p_random=warmup_p_random,
            warmup_p_heuristics=warmup_p_heuristics,
            start_mistake_rate=start_mistake_rate,
            final_mistake_rate=final_mistake_rate,
            p_random=p_random,
            p_heuristics=p_heuristics,
            local_mask_radius=local_mask_radius,
            mask_learner_until_steps=mask_learner_until_steps, # mask learner during warmup only
            mask_opponent_until_steps=mask_opponent_until_steps, # keep opponent local slightly longer than learner
            eval_games_per_side=eval_games_per_side,
            eval_seed=eval_seed,
            best_model_path=best_model_path,
            latest_model_path=latest_model_path,
            verbose=1,
        )

        model.learn(total_timesteps=total_timesteps, callback=self_play_cb)
        mlflow.log_params({
            "final_policy_state_sha256": policy_state_sha256(model),
            "final_num_timesteps": int(model.num_timesteps),
        })
        model.save(final_model_path)
        final_model_zip = f"{final_model_path}.zip"
        mlflow.log_artifact(final_model_zip, artifact_path="models")


if __name__ == "__main__":
    main()
