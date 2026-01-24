import time
import argparse
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from sb3_contrib import MaskablePPO
from x_in_a_row_sb3_env import SingleAgentSelfPlayEnv
from heuristic_policy import XInARowHeuristicPolicy

def test_model(env, model, n_episodes):
    for episode in range(n_episodes):
        obs, _info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0

        while not (terminated or truncated):
            action_masks = env.action_masks()
            action, _state = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(int(action))
            episode_reward += float(reward)
            env.render()

        print(f"Episode {episode + 1}/{n_episodes}: reward={episode_reward}")
        time.sleep(2) #Longer pause at end of game


def record_one_episode_gif(env, model, out_path: str, fps: int):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    obs, _info = env.reset()
    frame = env.render()
    if frame is not None:
        frames.append(np.asarray(frame))

    while True:
        # Safety: if episode is already over.
        if not env._env.agents:
            break

        # Learner/model move.
        action_masks = env.action_masks()
        action, _state = model.predict(obs, action_masks=action_masks, deterministic=True)
        env._env.step(int(action))

        # Frame after learner move.
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))

        if not env._env.agents:
            break

        # Opponent response(s) until it's learner's turn again.
        env._play_opponent_until_learner_turn()

        # Frame after opponent move(s).
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))

        if not env._env.agents:
            break

        # Update learner observation for next decision.
        obs = env._observe_for_learner()

    imageio.mimsave(str(out), frames, fps=int(fps))
    print(f"Wrote gif: {out}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gif", type=str, default=None)
    parser.add_argument("--fps", type=int, default=1)
    args = parser.parse_args()

    height = 3
    width = 3
    win_con = 3

    model = MaskablePPO.load("ppo_tic_tac_toe_baseline3")
    heuristic = XInARowHeuristicPolicy(height=height, width=width, win_con=win_con)

    env = SingleAgentSelfPlayEnv(
        height=height,
        width=width,
        win_con=win_con,
        p1_symbol="X",
        p2_symbol="O",
        render_mode=("rgb_array" if args.gif else "human"),
        opponent_policy=heuristic,
    )    

    if args.gif:
        env.set_opponent(model)
        record_one_episode_gif(env, model, args.gif, fps=args.fps)
        env.close()
        return

    print("Test 1: player 1 plays with model, player 2 plays with heuristic")
    test_model(env, model, 5)
    
    print("\nTest 2: player 1 plays with heuristic, player 2 plays with model")
    env.learner_symbol = "O"
    env.opponent_symbol = "X"
    
    test_model(env, model, 5)

    print("\nTest 3: both players play with model")
    env.learner_symbol = "X"
    env.opponent_symbol = "O"
    env.set_opponent(MaskablePPO.load("ppo_tic_tac_toe"))
    
    test_model(env, model, 2)
    env.close()


if __name__ == "__main__":
    main()
