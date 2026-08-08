import numpy as np
from x_in_a_row_env import XInARowEnv
from x_in_a_row_sb3_env import SingleAgentSelfPlayEnv
# Suppose you created the environment
env = SingleAgentSelfPlayEnv(
            height=15,
            width=15,
            win_con=5,
            p1_symbol="X",
            p2_symbol="O",
            render_mode=None,
            opponent_policy="random",
            randomize_learner=False,
            defensive_opening_prob = 1.0
        )
# Reset the environment
obs, info = env.reset()

done = False

# while env.agents:  # Loop until all agents have terminated or truncated
#agent = env.agent_selection

# Get observation for the current agent
obs = env._observe_for_learner()
print(obs)

# Select an action
# Example: random legal action
#action_mask = env.action_masks()
#print(action_mask)
# legal_actions = [i for i, valid in enumerate(action_mask) if valid]
# print(legal_actions)
# action = np.random.choice(legal_actions)
# # Step the environment
# env.step(action)
# env.render()

# # Get info for the current agent
# info = env.infos

# # Optional: print board for debugging
# print(f"Agent {agent} plays {action}")
# print(env.board)  # simplistic board view

# # After loop ends, you can access cumulative rewards
# print("Episode finished!")
# for agent in ["X", "O"]:
#     print(f"Agent {agent} cumulative reward: {env.cumulative_rewards[agent]}") 
