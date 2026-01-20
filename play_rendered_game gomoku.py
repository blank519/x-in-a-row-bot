import time
import numpy as np
from sb3_contrib import MaskablePPO
from x_in_a_row_sb3_env import SingleAgentSelfPlayEnv
from heuristic_policy import XInARowHeuristicPolicy, GomokuHeuristicPolicy

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

def main():
    height = 15
    width = 15
    win_con = 5

    #model = MaskablePPO.load("ppo_tic_tac_toe")
    heuristic = GomokuHeuristicPolicy()

    print("Test 1: player 1 plays with model, player 2 plays with heuristic")
    env = SingleAgentSelfPlayEnv(
        height=height,
        width=width,
        win_con=win_con,
        p1_symbol="X",
        p2_symbol="O",
        render_mode="human",
        opponent_policy=heuristic,
    )    

    for episode in range(5):
        obs, _info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0

        while not (terminated or truncated):
            action_masks = env.action_masks()
            action = heuristic(obs, action_masks, env.np_random)
            obs, reward, terminated, truncated, _info = env.step(int(action))
            episode_reward += float(reward)
            env.render()

        print(f"Episode {episode + 1}/{5}: reward={episode_reward}")
        time.sleep(2) #Longer pause at end of game


if __name__ == "__main__":
    main()
