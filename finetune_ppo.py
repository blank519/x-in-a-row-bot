from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib import MaskablePPO

# Import from correct file
from train_ppo_tic_tac_toe import BoardCnnExtractor, MaskableActorCriticPolicy, SelfPlaySnapshotCallback
from x_in_a_row_sb3_env import SingleAgentSelfPlayEnv


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

    base_model_path = "ppo_tic_tac_toe"
    finetuned_model_path = "ppo_tic_tac_toe_finetuned"

    snapshot_dir = "self_play_snapshots_finetune"
    snapshot_freq = 10_000

    n_envs = 8
    env = DummyVecEnv([make_env(height, width, win_con) for _ in range(n_envs)])

    model = MaskablePPO.load(
        base_model_path,
        env=env,
        custom_objects={"ent_coef": 0.001, "clip_range": 0.1, "learning_rate": 5e-5},
    )

    # Fine-tune: no random warmup; opponent mix is heuristic + snapshots only.
    # By setting p_random=0, once snapshots exist the pool will
    # automatically use snapshots as the remaining probability mass.
    self_play_cb = SelfPlaySnapshotCallback(
        vec_env=env,
        snapshot_dir=snapshot_dir,
        snapshot_freq=snapshot_freq,
        height=height,
        width=width,
        win_con=win_con,
        k=20,
        random_warmup_steps=0,
        mixed_warmup_steps=0,
        p_random=0.1,
        p_heuristic=0.7,
        verbose=1,
    )

    model.learn(total_timesteps=300_000, callback=self_play_cb)
    model.save(finetuned_model_path)


if __name__ == "__main__":
    main()
