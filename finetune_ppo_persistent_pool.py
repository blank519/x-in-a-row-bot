import os
import re

from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib import MaskablePPO

import mlflow

from train_ppo_gomoku import BoardCnnExtractor, MaskableActorCriticPolicy, SelfPlaySnapshotCallback
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


_SNAPSHOT_RE = re.compile(r"^opponent_snapshot_(\d+)\.zip$")


def load_snapshot_pool(snapshot_dir: str, k: int):
    if not os.path.isdir(snapshot_dir):
        return [], 0

    snapshots = []
    max_idx = 0
    for fname in os.listdir(snapshot_dir):
        m = _SNAPSHOT_RE.match(fname)
        if not m:
            continue
        idx = int(m.group(1))
        max_idx = max(max_idx, idx)
        stem = os.path.join(snapshot_dir, fname[:-4])
        snapshots.append((idx, stem))

    snapshots.sort(key=lambda t: t[0])
    if k is not None and k > 0:
        snapshots = snapshots[-k:]

    models = []
    for _idx, stem in snapshots:
        models.append(MaskablePPO.load(stem))

    return models, max_idx


def main():
    height = 15
    width = 15
    win_con = 5

    base_model_path = "outputs/ppo_gomoku_finetuned"
    finetuned_model_path = "outputs/ppo_gomoku_finetuned"

    snapshot_dir = "self_play_snapshots_finetune"
    snapshot_freq = 249_856

    n_envs = 8
    env = DummyVecEnv([make_env(height, width, win_con) for _ in range(n_envs)])

    k = 50
    num_timesteps = snapshot_freq * 10

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "ppo-gomoku-finetune-persistent"))

    with mlflow.start_run():
        mlflow.log_params({
            "base_model_path": base_model_path,
            "finetuned_model_path": finetuned_model_path,
            "height": height,
            "width": width,
            "win_con": win_con,
            "n_envs": n_envs,
            "snapshot_dir": snapshot_dir,
            "snapshot_freq": snapshot_freq,
            "num_timesteps": num_timesteps,
            "k": k,
            "random_warmup_steps": 0,
            "mixed_warmup_steps": 0,
            "mixed_p_random": 0.0,
            "mixed_p_heuristic": 0.0,
            "p_random": 0.1,
            "p_heuristics": "[0.2, 0.2]",
            "eval_games_per_side": 100,
        })

        model = MaskablePPO.load(
            base_model_path,
            env=env,
        )

        self_play_cb = SelfPlaySnapshotCallback(
            vec_env=env,
            snapshot_dir=snapshot_dir,
            snapshot_freq=snapshot_freq,
            height=height,
            width=width,
            win_con=win_con,
            k=k,
            random_warmup_steps=0,
            mixed_warmup_steps=0,
            mixed_p_random=0.0,
            mixed_p_heuristic=0.0,
            p_random=0.1,
            p_heuristics=[0.2, 0.2],
            eval_games_per_side=100,
            best_model_path="outputs/best_vs_heuristic_finetune",
            verbose=1,
        )

        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_models, max_idx = load_snapshot_pool(snapshot_dir=snapshot_dir, k=k)
        self_play_cb._snapshot_models = snapshot_models
        self_play_cb._snapshot_idx = max_idx
        mlflow.log_params({
            "loaded_snapshot_count": len(snapshot_models),
            "loaded_snapshot_max_idx": max_idx,
        })

        self_play_cb._best_saver.maybe_save(model, 0)

        model.learn(total_timesteps=num_timesteps, callback=self_play_cb)
        model.save(finetuned_model_path)
        mlflow.log_artifact(f"{finetuned_model_path}.zip", artifact_path="models")


if __name__ == "__main__":
    main()
