import os

import torch as th
import torch.nn as nn

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from x_in_a_row_sb3_env import SingleAgentSelfPlayEnv


class BoardCnnExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations.float()))


class SelfPlaySnapshotCallback(BaseCallback):
    def __init__(
        self,
        vec_env,
        snapshot_dir: str,
        snapshot_freq: int,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.vec_env = vec_env
        self.snapshot_dir = snapshot_dir
        self.snapshot_freq = int(snapshot_freq)
        self._snapshot_idx = 0

        os.makedirs(self.snapshot_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.snapshot_freq <= 0:
            return True

        if self.num_timesteps % self.snapshot_freq != 0:
            return True

        self._snapshot_idx += 1
        snapshot_path = f"{self.snapshot_dir}/opponent_snapshot_{self._snapshot_idx}"

        self.model.save(snapshot_path)
        opponent_model = MaskablePPO.load(snapshot_path)

        self.vec_env.env_method("set_opponent", opponent_model)

        if self.verbose > 0:
            print(f"[SelfPlay] Updated opponent from snapshot: {snapshot_path}.zip")

        return True


def make_env(height: int, width: int, win_con: int):
    def _thunk():
        return SingleAgentSelfPlayEnv(
            height=height,
            width=width,
            win_con=win_con,
            p1_symbol="X",
            p2_symbol="O",
            render_mode=None,
            opponent_policy="random",
            randomize_learner=True,
        )

    return _thunk


def main():
    height = 3
    width = 3
    win_con = 3

    snapshot_dir = "self_play_snapshots"
    snapshot_freq = 25_000

    n_envs = 8
    env = DummyVecEnv([make_env(height, width, win_con) for _ in range(n_envs)])

    model = MaskablePPO(
        policy=MaskableActorCriticPolicy,
        env=env,
        verbose=1,
        policy_kwargs={
            "features_extractor_class": BoardCnnExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
        },
        n_steps=64,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
    )

    self_play_cb = SelfPlaySnapshotCallback(
        vec_env=env,
        snapshot_dir=snapshot_dir,
        snapshot_freq=snapshot_freq,
        verbose=1,
    )

    model.learn(total_timesteps=500_000, callback=self_play_cb)
    model.save("ppo_tic_tac_toe")


if __name__ == "__main__":
    main()
