import os

from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib import MaskablePPO

import mlflow

from self_play_gomoku import BoardCnnExtractor, MaskableActorCriticPolicy, SelfPlaySnapshotCallback
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
    height = 15
    width = 15
    win_con = 5

    base_model_path = "outputs/ppo_gomoku"
    finetuned_model_path = "outputs/ppo_gomoku_finetuned"

    snapshot_dir = "self_play_snapshots_finetune"
    snapshot_freq = 249_424
    num_timesteps = 4_997_120

    n_envs = 8
    env = DummyVecEnv([make_env(height, width, win_con) for _ in range(n_envs)])

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "ppo-gomoku-finetune"))

    with mlflow.start_run():
        mlflow.log_params({
            "base_model_path": base_model_path,
            "finetuned_model_path": finetuned_model_path,
            "height": height,
            "width": width,
            "win_con": win_con,
            "n_envs": n_envs,
            "snapshot_freq": snapshot_freq,
            "num_timesteps": num_timesteps,
            "k": 20,
            "random_warmup_steps": 0,
            "mixed_warmup_steps": 0,
            "mixed_p_random": 0.0,
            "mixed_p_heuristic": 0.0,
            "p_random": 0.1,
            "p_heuristics": "[0.1, 0.25]",
            "eval_games_per_side": 100,
        })

        model = MaskablePPO.load(
            base_model_path,
            env=env,
            #custom_objects={"verbose":0, "ent_coef": 0.001, "clip_range": 0.1, "learning_rate": 5e-5},
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
            mixed_p_random=0.0,
            mixed_p_heuristic=0.0,
            p_random=0.1,
            p_heuristics=[0.1, 0.25],
            eval_games_per_side=100,
            best_model_path="outputs/best_vs_heuristic_finetune",
            verbose=1,
        )

        model.learn(total_timesteps=num_timesteps, callback=self_play_cb)
        model.save(finetuned_model_path)
        mlflow.log_artifact(f"{finetuned_model_path}.zip", artifact_path="models")


if __name__ == "__main__":
    main()
