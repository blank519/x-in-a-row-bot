"""Shared self-play training infrastructure for the Gomoku PPO agents.

This module holds the model/policy and self-play components that are reused
across the training and fine-tuning entry points (``train_ppo_gomoku.py``,
``finetune_ppo.py``, ``finetune_ppo_persistent_pool.py``). Keeping them here,
rather than inside a training script, means importing them does not execute a
training run.
"""

import os

import numpy as np

import torch as th
import torch.nn as nn

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from sb3_contrib import MaskablePPO
# Re-exported for convenience so callers can do `from self_play import
# MaskableActorCriticPolicy` alongside the other shared pieces.
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy  # noqa: F401

from x_in_a_row_sb3_env import apply_local_move_mask
from heuristic_policy import (
    XInARowHeuristicPolicy,
    GomokuDefensiveHeuristicPolicy,
    GomokuOffensiveHeuristicPolicy,
    GomokuCombinedHeuristicPolicy,
)
from vs_heuristic_eval import HeuristicEvaluator

import mlflow


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
        warmup_steps = 999_424,
        warmup_p_random = 0.3, # p(random) during warmup
        warmup_p_heuristics = [0.7], # p(heuristic) during warmup
        start_mistake_rate = 0.0, # Initial chance of combined heuristic making a mistake
        final_mistake_rate = 0.0, # Mistake rate at the end of the warmup anneal
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
        self.warmup_steps = warmup_steps
        self.warmup_p_random = warmup_p_random
        self.warmup_p_heuristics = warmup_p_heuristics
        self.start_mistake_rate = start_mistake_rate
        self.final_mistake_rate = final_mistake_rate
        self.local_mask_radius = local_mask_radius
        warmup_total = int(self.warmup_steps)
        self.mask_learner_until_steps = warmup_total if mask_learner_until_steps is None else int(mask_learner_until_steps)
        self.mask_opponent_until_steps = warmup_total if mask_opponent_until_steps is None else int(mask_opponent_until_steps)

        self._warmup_installed = False
        self._warmup_heuristic = None
        self._learner_mask_active = False
        self._learner_mask_removed = False
        self._opponent_mask_active = False
        self._opponent_mask_removed = False

        self.pool = OpponentPoolPolicy(
            height=height,
            width=width,
            win_con=win_con,
            p_random=p_random,
            p_heuristics=p_heuristics,
            # SET HEURISTICS HERE
            heuristics=[GomokuCombinedHeuristicPolicy(mistake_rate=start_mistake_rate)],
            local_move_radius=self.local_mask_radius,
            local_mask_enabled=False,
        )
        self._snapshot_models: list = []
        self._pool_installed = False

        # Running accumulators for the block-reward diagnostic, reset each
        # snapshot window. Lets us see whether the agent is actually blocking
        # (reducing opponent threat) independent of whole-game win rate.
        self._block_reward_sum = 0.0
        self._block_reward_steps = 0

        self.latest_model_path = latest_model_path
 
        self._best_saver = HeuristicEvaluator(
            height=height,
            width=width,
            win_con=win_con,
            heuristics=[GomokuCombinedHeuristicPolicy(),
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
        for info in self.locals.get("infos", []) or []:
            if "block_reward" in info:
                self._block_reward_sum += float(info["block_reward"])
                self._block_reward_steps += 1

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

        # Mask the opponent through the environment (same mechanism as the
        # learner), so every opponent type -- random, heuristic and snapshot --
        # plays under the same locality constraint. This replaces the old
        # OpponentPoolPolicy.local_mask_enabled self-masking.
        opponent_mask_should_be_on = (
            self.local_mask_radius is not None
            and self.num_timesteps < self.mask_opponent_until_steps
        )
        if opponent_mask_should_be_on and not self._opponent_mask_active:
            self.vec_env.env_method("set_opponent_local_mask_radius", self.local_mask_radius)
            self._opponent_mask_active = True
            self._opponent_mask_removed = False
        elif (not opponent_mask_should_be_on) and self._opponent_mask_active and not self._opponent_mask_removed:
            self.vec_env.env_method("set_opponent_local_mask_radius", None)
            self._opponent_mask_removed = True

        # Whether this step lands on an evaluation/snapshot boundary.
        on_snapshot_step = self.snapshot_freq > 0 and (self.num_timesteps % self.snapshot_freq == 0)

        # Warmup: opponent is a fixed mixture of random + combined heuristic
        # (no snapshots). The heuristic's mistake rate is annealed over the
        # warmup window.
        if self.num_timesteps < self.warmup_steps:
            if not self._warmup_installed:
                # Keep a reference to the combined heuristic so its mistake rate
                # can be annealed in place. DummyVecEnv shares this object across
                # all envs, so mutating it updates every environment's opponent.
                self._warmup_heuristic = GomokuCombinedHeuristicPolicy(mistake_rate=self.start_mistake_rate)
                warmup_opponent = OpponentPoolPolicy(
                    height=self.pool.height,
                    width=self.pool.width,
                    win_con=self.pool.win_con,
                    p_random=self.warmup_p_random,
                    p_heuristics=self.warmup_p_heuristics,
                    heuristics=[self._warmup_heuristic],
                    local_move_radius=self.local_mask_radius,
                    local_mask_enabled=False,  # opponent masking handled by the env
                )
                warmup_opponent.enable_heuristic(True)
                warmup_opponent.set_snapshots([])
                self.vec_env.env_method("set_opponent", warmup_opponent)
                self._warmup_installed = True

            # Linearly anneal the combined heuristic's mistake rate across the
            # warmup window: high early (beatable opponent -> positive reward
            # signal to learn blocking), decaying toward final_mistake_rate
            # (full-strength opponent) by the end of warmup.
            if self._warmup_heuristic is not None:
                progress = self.num_timesteps / max(1, self.warmup_steps)
                progress = min(1.0, max(0.0, progress))
                current_mistake_rate = self.start_mistake_rate + (self.final_mistake_rate - self.start_mistake_rate) * progress
                self._warmup_heuristic.set_mistake_rate(current_mistake_rate)
                if on_snapshot_step:
                    mlflow.log_metric("train/mistake_rate", float(current_mistake_rate), step=self.num_timesteps)

        else:
            # After warmup, enable heuristic in the main pool. Opponent locality
            # masking is applied by the env (set_opponent_local_mask_radius).
            self.pool.enable_heuristic(True)

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
            if self._block_reward_steps > 0:
                mlflow.log_metric(
                    "train/mean_block_reward",
                    self._block_reward_sum / self._block_reward_steps,
                    step=self.num_timesteps,
                )
            self._block_reward_sum = 0.0
            self._block_reward_steps = 0
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
