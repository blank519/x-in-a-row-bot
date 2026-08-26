import os
import re

import torch as th
import torch.nn as nn

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

from sb3_contrib import MaskablePPO

import mlflow
import datetime as dt

from self_play_gomoku import BoardCnnExtractor, MaskableActorCriticPolicy, SelfPlaySnapshotCallback
from x_in_a_row_sb3_env import SingleAgentSelfPlayEnv, CurriculumMaskedSelfPlayEnv

def make_env(height: int, width: int, win_con: int):
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
        )

    return _thunk
# def make_env(height: int, width: int, win_con: int):
#     def _thunk():
#         return SingleAgentSelfPlayEnv(
#             height=height,
#             width=width,
#             win_con=win_con,
#             p1_symbol="X",
#             p2_symbol="O",
#             render_mode=None,
#             opponent_policy="random",
#             randomize_learner=True,
#         )

#     return _thunk


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
    base_model_path = "outputs/ppo_gomoku_reproduce_og_results"
    finetuned_model_path = "outputs/ppo_gomoku_og_results_extended_training"

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

    # PPO parameters
    n_steps=512
    batch_size=512
    learning_rate= 1e-4 #lambda p: 1e-4 + p*(3e-4-1e-4) # p starts at 1 and goes to 0
    gamma=0.995
    gae_lambda=0.95
    ent_coef=0.005
    clip_range=0.1

    # Training schedule
    total_timesteps = 3_072_000  # Continuing heuristic warmup
    random_warmup_steps = 0
    mixed_warmup_steps = 3_072_000

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

    with mlflow.start_run(run_name=f"ppo-gomoku-og-results-extended-training-{dt.datetime.now().strftime('%Y-%m-%d-%H:%M')}"):
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
            "device": device,
        })

        model = MaskablePPO.load(
            base_model_path,
            env=env,
            device=device,
            seed=seed,
        )

        k=50
        self_play_cb = SelfPlaySnapshotCallback(
            vec_env=env,
            snapshot_dir=snapshot_dir,
            snapshot_freq=snapshot_freq,
            height=height,
            width=width,
            win_con=win_con,
            k=k, # max snapshot pool size
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

        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_models, max_idx = load_snapshot_pool(snapshot_dir=snapshot_dir, k=k)
        self_play_cb._snapshot_models = snapshot_models
        self_play_cb._snapshot_idx = max_idx
        mlflow.log_params({
            "loaded_snapshot_count": len(snapshot_models),
            "loaded_snapshot_max_idx": max_idx,
        })

        self_play_cb._best_saver.maybe_save(model, 0)

        model.learn(total_timesteps=total_timesteps, callback=self_play_cb)
        model.save(finetuned_model_path)
        mlflow.log_artifact(f"{finetuned_model_path}.zip", artifact_path="models")


if __name__ == "__main__":
    main()
