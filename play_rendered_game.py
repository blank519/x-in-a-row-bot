import time
import numpy as np
from sb3_contrib import MaskablePPO
from x_in_a_row_sb3_env import SingleAgentSelfPlayEnv


def main():
    height = 3
    width = 3
    win_con = 3

    env = SingleAgentSelfPlayEnv(
        height=height,
        width=width,
        win_con=win_con,
        learner_symbol="X",
        opponent_symbol="O",
        render_mode="human",
        opponent_policy="random",
    )

    model = MaskablePPO.load("ppo_x_in_a_row")

    n_episodes = 3
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
            time.sleep(0.5)

        print(f"Episode {episode + 1}/{n_episodes}: reward={episode_reward}")
        time.sleep(2) #Longer pause at end of game
    
    env.close()


if __name__ == "__main__":
    main()
