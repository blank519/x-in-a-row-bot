from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from x_in_a_row_sb3_env import SingleAgentSelfPlayEnv


def make_env(height: int, width: int, win_con: int):
    def _thunk():
        return SingleAgentSelfPlayEnv(
            height=height,
            width=width,
            win_con=win_con,
            learner_symbol="X",
            opponent_symbol="O",
            render_mode=None,
            opponent_policy="random",
        )

    return _thunk


def main():
    height = 3
    width = 3
    win_con = 3

    n_envs = 8
    env = DummyVecEnv([make_env(height, width, win_con) for _ in range(n_envs)])

    model = MaskablePPO(
        policy=MaskableActorCriticPolicy,
        env=env,
        verbose=1,
        n_steps=256,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
    )

    model.learn(total_timesteps=500_000)
    model.save("ppo_x_in_a_row")


if __name__ == "__main__":
    main()
