import os
import numpy as np

import torch as th
import torch.nn as nn

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from x_in_a_row_sb3_env import SingleAgentSelfPlayEnv
from heuristic_policy import XInARowHeuristicPolicy, GomokuDefensiveHeuristicPolicy, GomokuOffensiveHeuristicPolicy
from vs_heuristic_eval import HeuristicEvaluator

import mlflow

import datetime as dt

def apply_local_move_mask(
    action_mask: np.ndarray,
    obs: np.ndarray,
    height: int,
    width: int,
    radius: int | None,
    min_stones_before_mask: int = 1,
) -> np.ndarray:
    """Restrict legal moves to cells within Chebyshev radius of existing stones.

    Falls back to the original mask if the filtered mask is empty.
    """
    base_mask = np.asarray(action_mask, dtype=np.int8)
    if radius is None or radius < 0:
        return base_mask

    # Ignore local mask if the dimensions are too small
    obs_arr = np.asarray(obs)
    if obs_arr.ndim != 3 or obs_arr.shape[0] < 2:
        return base_mask

    # Ignore local mask if there are not enough stones on the board (by default < 1)
    occupied = (obs_arr[0] + obs_arr[1]) > 0
    if int(occupied.sum()) < int(min_stones_before_mask):
        return base_mask

    candidate = np.zeros((height, width), dtype=bool)
    occupied_coords = np.argwhere(occupied)
#    if occupied_coords.size == 0:
#        return base_mask

    for r, c in occupied_coords:
        r0 = max(0, int(r) - radius)
        r1 = min(height, int(r) + radius + 1)
        c0 = max(0, int(c) - radius)
        c1 = min(width, int(c) + radius + 1)
        candidate[r0:r1, c0:c1] = True

    masked = (base_mask.astype(bool) & candidate.reshape(-1))
    if masked.any():
        return masked.astype(np.int8)
    return base_mask

class BoardCnnExtractor(BaseFeaturesExtractor):
    # Requires expansion for Gomoku
    def __init__(self, observation_space, features_dim: int = 256):
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

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations.float()))


class OpponentPoolPolicy:
    def __init__(
        self,
        height,
        width,
        win_con,
        p_random = 0.25,
        p_heuristics = None,
        heuristics = None,
        local_move_radius: int | None = None,
        local_mask_enabled: bool = False,
    ):
        self.height = height
        self.width = width
        self.win_con = win_con

        self.p_random = p_random
        self.p_heuristics = list(p_heuristics) if p_heuristics is not None else [0.25]

        self.heuristic_enabled = False
        self.heuristics = list(heuristics) if heuristics is not None else [XInARowHeuristicPolicy(height, width, win_con)]
        if len(self.p_heuristics) != len(self.heuristics):
            raise ValueError(
                f"p_heuristics and heuristics must have the same length, got {len(self.p_heuristics)} and {len(self.heuristics)}"
            )
        self.snapshot_models: list = []
        self.local_move_radius = local_move_radius
        self.local_mask_enabled = bool(local_mask_enabled)

    def set_snapshots(self, snapshot_models):
        self.snapshot_models = list(snapshot_models)

    def enable_heuristic(self, enabled):
        self.heuristic_enabled = bool(enabled)

    def set_local_mask_enabled(self, enabled):
        self.local_mask_enabled = bool(enabled)

    def __call__(self, obs, action_mask, rng):
        mask = np.asarray(action_mask, dtype=np.int8)
        if self.local_mask_enabled:
            mask = apply_local_move_mask(
                action_mask=mask,
                obs=np.asarray(obs, dtype=np.int8),
                height=self.height,
                width=self.width,
                radius=self.local_move_radius,
            )
        legal = np.flatnonzero(mask.astype(bool)).astype(np.int64)
        if legal.size == 0:
            return 0

        available_snapshots = len(self.snapshot_models) > 0
        use_heuristic = self.heuristic_enabled

        p_random = self.p_random
        p_heuristic_total = float(sum(self.p_heuristics)) if use_heuristic else 0.0
        p_snap = max(0.0, 1.0 - (p_random + p_heuristic_total)) if available_snapshots else 0.0

        total = p_random + p_heuristic_total + p_snap

        r = float(rng.random()) * total
        if r < p_random:
            return int(rng.choice(legal))
        r -= p_random

        if use_heuristic:
            for i in range(len(self.p_heuristics)):
                if r < self.p_heuristics[i]:
                    return int(self.heuristics[i](obs, mask, rng))
                r -= self.p_heuristics[i]

        if available_snapshots and r <= p_snap:
            opponent_model = self.snapshot_models[int(rng.integers(0, len(self.snapshot_models)))]
            action, _state = opponent_model.predict(obs, action_masks=mask, deterministic=True)
            action = int(action)
            if mask[action] == 0:
                return int(rng.choice(legal))
            return action

        return int(rng.choice(legal))

class CurriculumMaskedSelfPlayEnv(SingleAgentSelfPlayEnv):
    """Single-agent self-play env with optional learner locality masking."""

    def __init__(self, *args, learner_local_mask_radius: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._learner_local_mask_radius = learner_local_mask_radius

    def set_learner_local_mask_radius(self, radius: int | None):
        self._learner_local_mask_radius = None if radius is None else int(radius)

    def action_masks(self) -> np.ndarray:
        base_mask = super().action_masks()
        return apply_local_move_mask(
            action_mask=base_mask,
            obs=self._env.observe(self.learner_symbol)["observation"],
            height=self.height,
            width=self.width,
            radius=self._learner_local_mask_radius,
        )


class SelfPlaySnapshotCallback(BaseCallback):
    def __init__(
        self,
        vec_env,
        snapshot_dir,
        snapshot_freq,
        height,
        width,
        win_con,
        k = 20,
        random_warmup_steps = 999_424,
        mixed_warmup_steps = 999_424,
        mixed_p_random = 0.3, # p(random) during mixed warmup
        mixed_p_heuristic = 0.7, # p(heuristic) during mixed warmup
        p_random = 0.1,
        p_heuristics = [0.2, 0.2], # p_random + p_heuristics should be <= 1. Remaining probability is snapshot pool.
        local_mask_radius: int | None = 2,
        mask_learner_until_steps: int | None = None,
        mask_opponent_until_steps: int | None = None,
        eval_games_per_side = 100,
        best_model_path = "best_vs_heuristic",
        latest_model_path = "outputs/latest_model",
        verbose = 0,
    ):
        super().__init__(verbose=verbose)
        self.vec_env = vec_env
        self.snapshot_dir = snapshot_dir
        self.snapshot_freq = snapshot_freq
        self._snapshot_idx = 0

        self.k = k
        self.random_warmup_steps = random_warmup_steps
        self.mixed_warmup_steps = mixed_warmup_steps
        self.mixed_p_random = mixed_p_random
        self.mixed_p_heuristic = mixed_p_heuristic
        self.local_mask_radius = local_mask_radius
        warmup_total = int(self.random_warmup_steps + self.mixed_warmup_steps)
        self.mask_learner_until_steps = warmup_total if mask_learner_until_steps is None else int(mask_learner_until_steps)
        self.mask_opponent_until_steps = warmup_total if mask_opponent_until_steps is None else int(mask_opponent_until_steps)

        self._random_warmup_installed = False
        self._mixed_warmup_installed = False
        self._learner_mask_active = False
        self._learner_mask_removed = False

        self.pool = OpponentPoolPolicy(
            height=height,
            width=width,
            win_con=win_con,
            p_random=p_random,
            p_heuristics=p_heuristics,
            # SET HEURISTICS HERE
            heuristics=[XInARowHeuristicPolicy(height=height, width=width, win_con=win_con), 
                        GomokuDefensiveHeuristicPolicy(),
                        GomokuOffensiveHeuristicPolicy(),
                        ],
            local_move_radius=self.local_mask_radius,
            local_mask_enabled=False,
        )
        self._snapshot_models: list = []
        self._pool_installed = False

        self.latest_model_path = latest_model_path
 
        self._best_saver = HeuristicEvaluator(
            height=height,
            width=width,
            win_con=win_con,
            heuristics=[XInARowHeuristicPolicy(height=height, width=width, win_con=win_con),
                        GomokuDefensiveHeuristicPolicy(),
                        GomokuOffensiveHeuristicPolicy()],
            n_games_per_side=eval_games_per_side,
            best_model_path=best_model_path,
            deterministic=True,
            seed=0,
            verbose=verbose,
        )

        os.makedirs(self.snapshot_dir, exist_ok=True)

    def _on_step(self) -> bool:
        learner_mask_should_be_on = (
            self.local_mask_radius is not None
            and self.num_timesteps < self.mask_learner_until_steps
        )
        if learner_mask_should_be_on and not self._learner_mask_active:
            self.vec_env.env_method("set_learner_local_mask_radius", self.local_mask_radius)
            self._learner_mask_active = True
            self._learner_mask_removed = False
        elif (not learner_mask_should_be_on) and self._learner_mask_active and not self._learner_mask_removed:
            self.vec_env.env_method("set_learner_local_mask_radius", None)
            self._learner_mask_removed = True

        opponent_mask_enabled = (
            self.local_mask_radius is not None
            and self.num_timesteps < self.mask_opponent_until_steps
        )

        # Whether this step lands on an evaluation/snapshot boundary.
        on_snapshot_step = self.snapshot_freq > 0 and (self.num_timesteps % self.snapshot_freq == 0)

        # Stage 1 warmup: opponent is purely random (no heuristic, no snapshots)
        if self.num_timesteps < self.random_warmup_steps:
            if not self._random_warmup_installed:
                warmup_opponent = OpponentPoolPolicy(
                    height=self.pool.height,
                    width=self.pool.width,
                    win_con=self.pool.win_con,
                    p_random=1.0,
                    p_heuristics=[0.0],
                    local_move_radius=self.local_mask_radius,
                    local_mask_enabled=opponent_mask_enabled,
                )
                warmup_opponent.enable_heuristic(False)
                warmup_opponent.set_snapshots([])
                self.vec_env.env_method("set_opponent", warmup_opponent)
                self._random_warmup_installed = True

        # Stage 2 warmup: opponent is a fixed mixture of random + heuristic (no snapshots)
        elif self.num_timesteps < (self.random_warmup_steps + self.mixed_warmup_steps):
            if not self._mixed_warmup_installed:
                warmup_opponent = OpponentPoolPolicy(
                    height=self.pool.height,
                    width=self.pool.width,
                    win_con=self.pool.win_con,
                    p_random=self.mixed_p_random,
                    # Split the heuristic mass between the weak baseline and the
                    # offensive attacker so the learner starts seeing threats early.
                    p_heuristics=[self.mixed_p_heuristic * 0.4, self.mixed_p_heuristic * 0.25, self.mixed_p_heuristic * 0.35],
                    heuristics=[XInARowHeuristicPolicy(height=self.pool.height, width=self.pool.width, win_con=self.pool.win_con),
                                GomokuDefensiveHeuristicPolicy(),
                                GomokuOffensiveHeuristicPolicy()],
                    local_move_radius=self.local_mask_radius,
                    local_mask_enabled=opponent_mask_enabled,
                )
                warmup_opponent.enable_heuristic(True)
                warmup_opponent.set_snapshots([])
                self.vec_env.env_method("set_opponent", warmup_opponent)
                self._mixed_warmup_installed = True

        else:
            # After warmup, enable heuristic in the main pool.
            self.pool.enable_heuristic(True)
            self.pool.set_local_mask_enabled(opponent_mask_enabled)

            if not self._pool_installed:
                self.pool.set_snapshots(self._snapshot_models)
                self.vec_env.env_method("set_opponent", self.pool)
                self._pool_installed = True

            # Freeze a snapshot into the opponent pool on snapshot boundaries.
            if on_snapshot_step:
                self._snapshot_idx += 1
                snapshot_path = f"{self.snapshot_dir}/opponent_snapshot_{self._snapshot_idx}"

                self.model.save(snapshot_path)
                opponent_model = MaskablePPO.load(snapshot_path)

                self._snapshot_models.append(opponent_model)
                if len(self._snapshot_models) > self.k:
                    self._snapshot_models = self._snapshot_models[-self.k :]

                self.pool.set_snapshots(self._snapshot_models)
                self.vec_env.env_method("set_opponent", self.pool)

                if self.verbose > 0:
                    print(f"[SelfPlay] Updated opponent from snapshot: {snapshot_path}.zip")

        # Evaluate and save the most recent model on every snapshot boundary,
        # regardless of which training stage we are currently in.
        if on_snapshot_step:
            self._evaluate_and_save()

        return True

    def _evaluate_and_save(self) -> None:
        # Always persist the most recent model so the latest checkpoint is
        # available even when it is not the best-scoring one.
        if self.latest_model_path:
            os.makedirs(os.path.dirname(self.latest_model_path) or ".", exist_ok=True)
            self.model.save(self.latest_model_path)

        # Evaluate against the heuristics and save the best model if improved.
        improved, metrics = self._best_saver.maybe_save(self.model, self.num_timesteps)
        mlflow.log_metrics({f"eval/{k}": float(v) for k, v in metrics.items()}, step=self.num_timesteps)
        mlflow.log_metric("eval/improved", int(improved), step=self.num_timesteps)


def make_env(height: int, width: int, win_con: int,
             reward_shaping_coef: float = 0.0, reward_shaping_gamma: float = 0.99):
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
        )

    return _thunk


def main():
    height = 15
    width = 15
    win_con = 5

    seed = 42
    set_random_seed(seed, using_cuda=th.cuda.is_available())

    snapshot_dir = "self_play_snapshots"
    snapshot_freq = 256_000 # Close to 250k, multiple of batch_size * n_environments

    # Potential-based reward shaping to give a dense signal for building threats
    # and blocking the opponent. Set to 0.0 to disable. reward_shaping_gamma
    # should match the PPO gamma below for policy-invariant shaping.
    reward_shaping_coef = 0.01
    reward_shaping_gamma = 0.995

    n_envs = 16
    env = DummyVecEnv([
        make_env(
            height, width, win_con,
            reward_shaping_coef=reward_shaping_coef,
            reward_shaping_gamma=reward_shaping_gamma,
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

    # PPO parameters
    n_steps=512
    batch_size=512
    learning_rate= 1e-4 #lambda p: 1e-4 + p*(3e-4-1e-4) # p starts at 1 and goes to 0
    gamma=0.995
    gae_lambda=0.95
    ent_coef=0.005
    clip_range=0.1

    # Training schedule
    total_timesteps = 8_192_000  # Compare results with finetune_ppo_persistent_pool.py
    random_warmup_steps = 2_048_000
    mixed_warmup_steps = 6_144_000 #3_072_000

    # Opponent pool parameters
    mixed_p_random = 0.3
    mixed_p_heuristic = 0.7
    p_random = 0.1
    # [weak, defensive (block-3s), offensive (build-5s)]; remainder -> snapshot pool
    p_heuristics = [0.1, 0.2, 0.2]
    local_mask_radius = 2
    mask_learner_until_steps = random_warmup_steps + mixed_warmup_steps  # mask learner during warmup only
    mask_opponent_until_steps = mask_learner_until_steps + 0  # keep opponent local slightly longer than learner
    eval_games_per_side = 100

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ppo-gomoku")

    with mlflow.start_run(run_name=f"ppo-gomoku-reward-shaping-{dt.datetime.now().strftime('%Y-%m-%d-%H:%M')}"):
        mlflow.log_params({
            "n_envs": n_envs,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "ent_coef": ent_coef,
            "clip_range": clip_range,
            "snapshot_freq": snapshot_freq,
            "random_warmup_steps": random_warmup_steps, # ~2M for tactical bootstrapping with random play
            "mixed_warmup_steps": mixed_warmup_steps, # ~8M for guided play before full self-play
            "mixed_p_random": mixed_p_random,
            "mixed_p_heuristic": mixed_p_heuristic,
            "p_random": p_random,
            "p_heuristics": p_heuristics,
            "local_mask_radius": local_mask_radius,
            "mask_learner_until_steps": mask_learner_until_steps, # mask learner during warmup only
            "mask_opponent_until_steps": mask_opponent_until_steps, # keep opponent local slightly longer than learner
            "eval_games_per_side": eval_games_per_side,
            "total_timesteps": total_timesteps, # Entire run will be warmup
            "reward_shaping_coef": reward_shaping_coef,
            "reward_shaping_gamma": reward_shaping_gamma,
            "device": device,
        })

        model = MaskablePPO(
            policy=MaskableActorCriticPolicy,
            env=env,
            verbose=1,
            policy_kwargs={
                "features_extractor_class": BoardCnnExtractor,
                "features_extractor_kwargs": {"features_dim": 512},
            },
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
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
            random_warmup_steps=random_warmup_steps, # ~2M for tactical bootstrapping with random play
            mixed_warmup_steps=mixed_warmup_steps, # ~3M for guided play before full self-play
            mixed_p_random=mixed_p_random,
            mixed_p_heuristic=mixed_p_heuristic,
            p_random=p_random,
            p_heuristics=p_heuristics,
            local_mask_radius=local_mask_radius,
            mask_learner_until_steps=mask_learner_until_steps, # mask learner during warmup only
            mask_opponent_until_steps=mask_opponent_until_steps, # keep opponent local slightly longer than learner
            eval_games_per_side=eval_games_per_side,
            best_model_path="outputs/best_vs_heuristic",
            latest_model_path="outputs/latest_vs_heuristic",
            verbose=1,
        )

        model.learn(total_timesteps=total_timesteps, callback=self_play_cb)
        model.save("outputs/ppo_gomoku_reward_shaping")
        final_model_zip = "outputs/ppo_gomoku_reward_shaping.zip"
        mlflow.log_artifact(final_model_zip, artifact_path="models")


if __name__ == "__main__":
    main()
