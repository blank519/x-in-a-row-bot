# Unit: LRCONST-S43 (ticket reestablish_baseline, attempt 2, plan_2.md).
# Single-mechanism iteration derived from this seed's retained attempt-1
# control (MLflow run 982ce99ca17b466db62a86ac5efb9601,
# "ppo-gomoku-rebaseline-ent010-seed43-2026-11-07"), which itself
# reconstructs the immutable pre-fix diagnostic champion
# (MLflow run b9299ce03cdf46da8b0f9b969396c15f,
# "ppo-gomoku-warmup-wideproj1024-entropy010-block15-15m-2026-10-14",
# experiment ppo-gomoku). The attempt-1 scientific configuration is
# preserved exactly; the ONLY scientific delta versus the matched attempt-1
# control is the learning-rate schedule: the historical rising expression
# `lambda p: 3e-4 + p*(1e-4-3e-4)` (SB3 progress_remaining p: 1 -> 0, so it
# actually RISES 1e-4 -> 3e-4) is replaced by the scalar constant
# `learning_rate = 1e-4`, removing the late optimization pressure of the
# rise to 3e-4. Entropy stays 0.010 and no entropy callback is introduced.
# Hypothesis: stabilizing late updates will allow seed 43 to generalize its
# fast Offensive-X tactic into Combined/Defensive/O-side play instead of
# collapsing into the mature Offensive-X specialization. All post-fix
# reproducibility implementation details are unchanged: strict
# configure_reproducibility(seed) before environment creation or any CUDA
# query, an eval-mode zero-tensor shape probe in the script-local wide
# extractor (never observation_space.sample(), never mutating BatchNorm
# statistics), eval_seed=0, provenance-first log_reproducibility_metadata
# inside mlflow.start_run, and the final semantic policy_state_sha256 /
# final timestep params. Run name is static; do not rename. This script is
# the unit's reproducible launch configuration and must be retained after
# evaluation.

import torch as th
import torch.nn as nn

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from sb3_contrib import MaskablePPO

from x_in_a_row_sb3_env import CurriculumMaskedSelfPlayEnv
from self_play_gomoku import MaskableActorCriticPolicy, SelfPlaySnapshotCallback
from reproducibility import configure_reproducibility, log_reproducibility_metadata, policy_state_sha256

import mlflow


class WideProjectionBoardCnnExtractor(BaseFeaturesExtractor):
    """Script-local deterministic wide-projection feature extractor.

    Script-local variant of the current (post-fix) ``BoardCnnExtractor``:
    the identical five-convolution stack ([64, 64, 128, 128, 128] channels,
    3x3 kernels, padding 1, BatchNorm/ReLU order) and flattening, then the
    champion's widened post-flatten projection Linear(n_flatten, 1024) ->
    ReLU -> Linear(1024, features_dim) -> ReLU. The shape probe uses a zero
    tensor in eval mode: it neither consumes ``observation_space.sample()``
    nor mutates BatchNorm running statistics, so same-seed constructions are
    bit-identical. No architectural change relative to the immutable
    baseline's wide extractor.
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

        # Deterministic shape probe: zero tensor, eval mode, no-grad. Keeps
        # BatchNorm running statistics at their initial values and does not
        # touch the observation-space RNG.
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
    run_name = "ppo-gomoku-rebaseline-lrconst1e4-ent010-seed43-2026-11-08"
    unit_id = "LRCONST-S43"
    ticket = "reestablish_baseline"
    attempt = 2
    baseline_run_id = "b9299ce03cdf46da8b0f9b969396c15f"
    baseline_run_name = "ppo-gomoku-warmup-wideproj1024-entropy010-block15-15m-2026-10-14"
    # Matched same-seed attempt-1 control (immutable; read-only reference).
    control_run_id = "982ce99ca17b466db62a86ac5efb9601"
    control_run_name = "ppo-gomoku-rebaseline-ent010-seed43-2026-11-07"
    control_script_path = "train_ppo_gomoku_ppo-gomoku-rebaseline-ent010-seed43-2026-11-07.py"
    script_path = "train_ppo_gomoku_ppo-gomoku-rebaseline-lrconst1e4-ent010-seed43-2026-11-08.py"
    historical_source = (
        "artifacts/raise_warmup_performance_4/"
        "train_ppo_gomoku_ppo-gomoku-warmup-wideproj1024-entropy010-block15-15m-2026-10-14.py"
    )

    height = 15
    width = 15
    win_con = 5

    seed = 43  # Attempt-2 identity: same-seed as the matched attempt-1 control.
    eval_seed = 0

    # Strict reproducibility must precede environment creation and any CUDA
    # query/context use (post-fix requirement).
    configure_reproducibility(seed)

    # Run-specific destinations (disjoint from all sibling units).
    snapshot_dir = "self_play_snapshots/ppo-gomoku-rebaseline-lrconst1e4-ent010-seed43-2026-11-08"
    best_model_path = "outputs/best_vs_heuristic_ppo-gomoku-rebaseline-lrconst1e4-ent010-seed43-2026-11-08"
    latest_model_path = "outputs/latest_vs_heuristic_ppo-gomoku-rebaseline-lrconst1e4-ent010-seed43-2026-11-08"
    final_model_path = "outputs/ppo-gomoku-rebaseline-lrconst1e4-ent010-seed43-2026-11-08"
    snapshot_freq = 256_000  # Close to 250k, multiple of batch_size * n_environments

    # Immutable baseline configuration (b9299ce03cdf46da8b0f9b969396c15f).
    # Potential-based reward shaping disabled, matching the baseline.
    reward_shaping_coef = 0.000
    reward_shaping_gamma = 0.995
    # Immediate block reward, baseline value 0.15.
    block_reward_coef = 0.15
    # Defensive-opening curriculum, baseline value 0.3.
    defensive_opening_prob = 0.3

    # PPO parameters (attempt-1 control values).
    n_steps = 512
    batch_size = 512
    # Attempt-2 single-mechanism delta: the historical rising LR expression
    # `lambda p: 3e-4 + p*(1e-4-3e-4)` (SB3 progress_remaining p: 1 -> 0, so
    # it rises 1e-4 -> 3e-4) is replaced by the scalar constant `1e-4`,
    # removing the late optimization pressure of the rise to 3e-4. Entropy,
    # clip, and every other knob remain the attempt-1 control values.
    learning_rate = 1e-4
    gamma = 0.995
    gae_lambda = 0.95
    ent_coef = 0.010  # Baseline (entropy010) value, unchanged.
    clip_range = 0.1

    # Training schedule: entire run is warmup, ending exactly at the warmup
    # boundary (baseline values). The four-way equality below is mandatory and
    # prohibits opponent-pool training; the shared callback may create one
    # isolated snapshot exactly at the equality boundary.
    total_timesteps = 15_360_000
    warmup_steps = 15_360_000

    # Opponent pool parameters (baseline values).
    # For warmup: p_random + sum(p_heuristics) + p_snapshot should equal 1.0
    warmup_p_random = 0.1
    warmup_p_heuristics = [0.9]  # Combined heuristic
    start_mistake_rate = 0.7  # Initial chance of combined heuristic making a mistake
    final_mistake_rate = 0.1  # Mistake rate at the end of the warmup anneal

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
        f"unit_id={unit_id} ticket={ticket} attempt={attempt} "
        f"unit_delta=learning_rate_rising_1e-4_to_3e-4_to_constant_1e-4 "
        f"learning_rate_schedule=constant_1e-4 "
        f"control={control_run_id} "
        f"baseline={baseline_run_id}"
    )

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ppo-gomoku")

    with mlflow.start_run(run_name=run_name):
        # Provenance must be the first operation inside the active run.
        log_reproducibility_metadata(
            seed=seed,
            eval_seed=eval_seed,
            device=device,
            entrypoint_path=__file__,
            input_artifacts=(control_script_path, historical_source),
        )

        mlflow.log_params({
            # Identity / provenance
            "run_name": run_name,
            "unit_id": unit_id,
            "ticket": ticket,
            "attempt": attempt,
            "unit_delta": "learning_rate_rising_1e-4_to_3e-4_to_constant_1e-4",
            "baseline_run_id": baseline_run_id,
            "baseline_run_name": baseline_run_name,
            "control_run_id": control_run_id,
            "control_run_name": control_run_name,
            "control_script_path": control_script_path,
            "script_source": "train_ppo_gomoku.py",
            "script_path": script_path,
            # Architecture (baseline reconstruction; unchanged in this unit)
            "feature_extractor": "WideProjectionBoardCnnExtractor",
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
            "warmup_steps": warmup_steps,  # guided play before full self-play
            "warmup_p_random": warmup_p_random,
            "warmup_p_heuristics": warmup_p_heuristics,
            "start_mistake_rate": start_mistake_rate,
            "final_mistake_rate": final_mistake_rate,
            "p_random": p_random,
            "p_heuristics": p_heuristics,
            "local_mask_radius": local_mask_radius,
            "mask_learner_until_steps": mask_learner_until_steps,  # mask learner during warmup only
            "mask_opponent_until_steps": mask_opponent_until_steps,  # keep opponent local slightly longer than learner
            "eval_games_per_side": eval_games_per_side,
            "total_timesteps": total_timesteps,  # Entire run will be warmup
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
                # Baseline reconstruction: wire this script's policy_kwargs to
                # the script-local deterministic wide-projection extractor
                # (1024 hidden, features_dim 512); no architectural delta.
                "features_extractor_class": WideProjectionBoardCnnExtractor,
                "features_extractor_kwargs": {"features_dim": 512},
            },
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,  # Attempt-2 delta: scalar constant 1e-4.
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
            k=50,  # max snapshot pool size
            warmup_steps=warmup_steps,  # guided play before full self-play
            warmup_p_random=warmup_p_random,
            warmup_p_heuristics=warmup_p_heuristics,
            start_mistake_rate=start_mistake_rate,
            final_mistake_rate=final_mistake_rate,
            p_random=p_random,
            p_heuristics=p_heuristics,
            local_mask_radius=local_mask_radius,
            mask_learner_until_steps=mask_learner_until_steps,  # mask learner during warmup only
            mask_opponent_until_steps=mask_opponent_until_steps,  # keep opponent local slightly longer than learner
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
