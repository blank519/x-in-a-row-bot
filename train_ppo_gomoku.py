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

from x_in_a_row_sb3_env import SingleAgentSelfPlayEnv, CurriculumMaskedSelfPlayEnv, apply_local_move_mask
from heuristic_policy import XInARowHeuristicPolicy, GomokuDefensiveHeuristicPolicy, GomokuOffensiveHeuristicPolicy, GomokuCombinedHeuristicPolicy
from vs_heuristic_eval import HeuristicEvaluator

import mlflow

import datetime as dt

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
        random_warmup_steps = 999_424,
        mixed_warmup_steps = 999_424,
        mixed_p_random = 0.3, # p(random) during mixed warmup
        mixed_p_heuristics = [0.7], # p(heuristic) during mixed warmup
        start_mistake_rate = 0.0, # Initial chance of combined heuristic making a mistake
        final_mistake_rate = 0.0, # Mistake rate at the end of the mixed-warmup anneal
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
        self.mixed_p_heuristics = mixed_p_heuristics
        self.start_mistake_rate = start_mistake_rate
        self.final_mistake_rate = final_mistake_rate
        self.local_mask_radius = local_mask_radius
        warmup_total = int(self.random_warmup_steps + self.mixed_warmup_steps)
        self.mask_learner_until_steps = warmup_total if mask_learner_until_steps is None else int(mask_learner_until_steps)
        self.mask_opponent_until_steps = warmup_total if mask_opponent_until_steps is None else int(mask_opponent_until_steps)

        self._random_warmup_installed = False
        self._mixed_warmup_installed = False
        self._mixed_heuristic = None
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
                    local_mask_enabled=False,  # opponent masking handled by the env
                )
                warmup_opponent.enable_heuristic(False)
                warmup_opponent.set_snapshots([])
                self.vec_env.env_method("set_opponent", warmup_opponent)
                self._random_warmup_installed = True

        # Stage 2 warmup: opponent is a fixed mixture of random + heuristic (no snapshots)
        elif self.num_timesteps < (self.random_warmup_steps + self.mixed_warmup_steps):
            if not self._mixed_warmup_installed:
                # Keep a reference to the combined heuristic so its mistake rate
                # can be annealed in place. DummyVecEnv shares this object across
                # all envs, so mutating it updates every environment's opponent.
                self._mixed_heuristic = GomokuCombinedHeuristicPolicy(mistake_rate=self.start_mistake_rate)
                warmup_opponent = OpponentPoolPolicy(
                    height=self.pool.height,
                    width=self.pool.width,
                    win_con=self.pool.win_con,
                    p_random=self.mixed_p_random,
                    p_heuristics=self.mixed_p_heuristics,
                    heuristics=[self._mixed_heuristic],
                    local_move_radius=self.local_mask_radius,
                    local_mask_enabled=False,  # opponent masking handled by the env
                )
                warmup_opponent.enable_heuristic(True)
                warmup_opponent.set_snapshots([])
                self.vec_env.env_method("set_opponent", warmup_opponent)
                self._mixed_warmup_installed = True

            # Linearly anneal the combined heuristic's mistake rate across the
            # mixed-warmup window: high early (beatable opponent -> positive
            # reward signal to learn blocking), decaying toward final_mistake_rate
            # (full-strength opponent) by the end of warmup.
            if self._mixed_heuristic is not None:
                progress = (self.num_timesteps - self.random_warmup_steps) / max(1, self.mixed_warmup_steps)
                progress = min(1.0, max(0.0, progress))
                current_mistake_rate = self.start_mistake_rate + (self.final_mistake_rate - self.start_mistake_rate) * progress
                self._mixed_heuristic.set_mistake_rate(current_mistake_rate)
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


def make_env(height: int, width: int, win_con: int,
             reward_shaping_coef: float = 0.0, reward_shaping_gamma: float = 0.99,
             reward_shaping_defense_weight: float = 1.0,
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
            reward_shaping_defense_weight=reward_shaping_defense_weight,
            block_reward_coef=block_reward_coef,
            defensive_opening_prob=defensive_opening_prob,
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
    reward_shaping_coef = 0.000
    reward_shaping_gamma = 0.995
    # >1 rewards blocking the opponent more than building own threats. The model
    # currently only attacks (see f7caf4ff), so bias the potential toward defense.
    reward_shaping_defense_weight = 1.0
    # Immediate (non-potential) reward for reducing the opponent's threat mass with
    # the learner's own move. Unlike potential shaping this DOES change the optimal
    # policy, so it directly incentivizes blocking. A decisive "block the four"
    # move reduces threat mass by ~0.6-1.3, so coef ~0.5 makes a good block worth a
    # meaningful fraction of a terminal win. 0.0 disables it.
    block_reward_coef = 0.4
    # Fraction of training episodes that start from a designed "block-or-lose"
    # position (opponent has an open threat, learner to move). Directly teaches
    # defense, which sparse self-play never reaches. 0.0 disables the curriculum.
    defensive_opening_prob = 0.3

    # PPO parameters
    n_steps=512
    batch_size=512
    start_learning_rate=3e-4 # Default: 1e-4
    final_learning_rate=1e-4  
    gamma=0.995
    gae_lambda=0.95
    ent_coef=0.005
    clip_range=0.1

    # Training schedule
    total_timesteps = 10_240_000  # Compare results with finetune_ppo_persistent_pool.py
    random_warmup_steps = 0 #Attempt 0-random-warmup training (normally 2_048_000)
    mixed_warmup_steps = 10_240_000 #Last value 8_192_000

    # Opponent pool parameters
    # For mixed warmup: p_random + sum(p_heuristics) + p_snapshot should equal 1.0
    mixed_p_random = 0.1
    mixed_p_heuristics = [0.9] # Combined heuristic
    start_mistake_rate = 0.7 # Initial chance of combined heuristic making a mistake
    final_mistake_rate = 0.1 # Mistake rate at the end of the mixed-warmup anneal

    # For self-play: p_random + sum(p_heuristics) < 1.0, remainder = snapshot pool (CURRENTLY UNUSED)
    p_random = 0.1
    p_heuristics = [0.4]
    local_mask_radius = 2
    mask_learner_until_steps = random_warmup_steps + mixed_warmup_steps  # mask learner during warmup only
    mask_opponent_until_steps = mask_learner_until_steps + 0  # keep opponent local slightly longer than learner
    eval_games_per_side = 100

    n_envs = 16
    env = DummyVecEnv([
        make_env(
            height, width, win_con,
            reward_shaping_coef=reward_shaping_coef,
            reward_shaping_gamma=reward_shaping_gamma,
            reward_shaping_defense_weight=reward_shaping_defense_weight,
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


    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ppo-gomoku")

    with mlflow.start_run(run_name=f"ppo-gomoku-block-reward-sweep-40p-{dt.datetime.now().strftime('%Y-%m-%d-%H:%M')}"):
        mlflow.log_params({
            "n_envs": n_envs,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "starting_learning_rate": start_learning_rate,
            "final_learning_rate": final_learning_rate,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "ent_coef": ent_coef,
            "clip_range": clip_range,
            "snapshot_freq": snapshot_freq,
            "random_warmup_steps": random_warmup_steps, # ~2M for tactical bootstrapping with random play
            "mixed_warmup_steps": mixed_warmup_steps, # ~8M for guided play before full self-play
            "mixed_p_random": mixed_p_random,
            "mixed_p_heuristics": mixed_p_heuristics,
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
            "reward_shaping_defense_weight": reward_shaping_defense_weight,
            "block_reward_coef": block_reward_coef,
            "defensive_opening_prob": defensive_opening_prob,
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
            learning_rate=lambda p: start_learning_rate + p*(final_learning_rate-start_learning_rate),
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
            mixed_p_heuristics=mixed_p_heuristics,
            start_mistake_rate=start_mistake_rate,
            final_mistake_rate=final_mistake_rate,
            p_random=p_random,
            p_heuristics=p_heuristics,
            local_mask_radius=local_mask_radius,
            mask_learner_until_steps=mask_learner_until_steps, # mask learner during warmup only
            mask_opponent_until_steps=mask_opponent_until_steps, # keep opponent local slightly longer than learner
            eval_games_per_side=eval_games_per_side,
            best_model_path=f"outputs/best_vs_heuristic_block_reward_sweep_40p",
            latest_model_path=f"outputs/latest_vs_heuristic_block_reward_sweep_40p",
            verbose=1,
        )

        model.learn(total_timesteps=total_timesteps, callback=self_play_cb)
        model.save(f"outputs/ppo_gomoku_block_reward_sweep_40p")
        final_model_zip = f"outputs/ppo_gomoku_block_reward_sweep_40p.zip"
        mlflow.log_artifact(final_model_zip, artifact_path="models")


if __name__ == "__main__":
    main()
