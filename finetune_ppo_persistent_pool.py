import os
import re

import torch as th
import torch.nn as nn

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

from sb3_contrib import MaskablePPO

import mlflow
import datetime as dt

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

    seed = 42
    set_random_seed(seed, using_cuda = th.cuda.is_available())
    base_model_path = "outputs/ppo_gomoku_more_heuristic"
    finetuned_model_path = "outputs/ppo_gomoku_more_heuristic_extended"

    snapshot_dir = "self_play_snapshots"
    snapshot_freq = 256_000

    n_envs = 16
    env = DummyVecEnv([make_env(height, width, win_con) for _ in range(n_envs)])
    env.seed(seed)
    device = "cuda" if th.cuda.is_available() else "cpu"

    print(
        f"[Train] device={device} "
        f"cuda_available={th.cuda.is_available()} "
        f"torch_version={th.__version__}"
    )

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ppo-gomoku")

    with mlflow.start_run(run_name=f"ppo-gomoku-{dt.datetime.now().strftime('%Y-%m-%d-%H:%M')}-extended-training"):
        mlflow.log_params({
            "n_envs": n_envs,
            "n_steps": 64,
            "batch_size": 256,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
            "clip_range": 0.2,
            "snapshot_freq": snapshot_freq,
            "random_warmup_steps": 2_048_000, # ~2M for tactical bootstrapping with random play
            "mixed_warmup_steps": 3_072_000, # ~3M for guided play before full self-play
            "mixed_p_random": 0.3,
            "mixed_p_heuristic": 0.7,
            "p_random": 0.1,
            "p_heuristics": [0.1, 0.2, 0.2],
            "local_mask_radius": 2,
            "mask_learner_until_steps": 5_120_000, # mask learner during warmup only
            "mask_opponent_until_steps": 6_144_000, # keep opponent local slightly longer than learner
            "eval_games_per_side": 100,
            "total_timesteps": 20_480_000,
            "device": device,
        })

        model = MaskablePPO.load(
            base_model_path,
            env=env,
            device=device,
            seed=seed,
        )

        k = 50 # max snapshot pool size
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
            p_heuristics=[0.1, 0.2, 0.2],
            local_mask_radius=2,
            eval_games_per_side=100,
            best_model_path="outputs/best_vs_heuristic_extended_training",
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

        model.learn(total_timesteps=10_240_000, callback=self_play_cb)
        model.save(finetuned_model_path)
        mlflow.log_artifact(f"{finetuned_model_path}.zip", artifact_path="models")


if __name__ == "__main__":
    main()
