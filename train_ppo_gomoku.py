import torch as th

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

from sb3_contrib import MaskablePPO

from x_in_a_row_sb3_env import CurriculumMaskedSelfPlayEnv
from self_play_gomoku import BoardCnnExtractor, MaskableActorCriticPolicy, SelfPlaySnapshotCallback

import mlflow

import datetime as dt


def make_env(height: int, width: int, win_con: int,
             reward_shaping_coef: float = 0.0, reward_shaping_gamma: float = 0.99,
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
    warmup_steps = 10_240_000 #Last value 8_192_000

    # Opponent pool parameters
    # For warmup: p_random + sum(p_heuristics) + p_snapshot should equal 1.0
    warmup_p_random = 0.1
    warmup_p_heuristics = [0.9] # Combined heuristic
    start_mistake_rate = 0.7 # Initial chance of combined heuristic making a mistake
    final_mistake_rate = 0.1 # Mistake rate at the end of the warmup anneal

    # For self-play: p_random + sum(p_heuristics) < 1.0, remainder = snapshot pool (CURRENTLY UNUSED)
    p_random = 0.1
    p_heuristics = [0.4]
    local_mask_radius = 2
    mask_learner_until_steps = warmup_steps  # mask learner during warmup only
    mask_opponent_until_steps = mask_learner_until_steps + 0  # keep opponent local slightly longer than learner
    eval_games_per_side = 100

    n_envs = 16
    env = DummyVecEnv([
        make_env(
            height, width, win_con,
            reward_shaping_coef=reward_shaping_coef,
            reward_shaping_gamma=reward_shaping_gamma,
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
            "warmup_steps": warmup_steps, # guided play before full self-play
            "warmup_p_random": warmup_p_random,
            "warmup_p_heuristics": warmup_p_heuristics,
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
            warmup_steps=warmup_steps, # guided play before full self-play
            warmup_p_random=warmup_p_random,
            warmup_p_heuristics=warmup_p_heuristics,
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
