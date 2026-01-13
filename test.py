import numpy as np
from x_in_a_row_env import XInARowEnv
# Suppose you created the environment
env = XInARowEnv(height=3, width=3, win_con=3, p1="X", p2="O", render_mode="human")

# Reset the environment
obs, info = env.reset()

done = False

while env.agents:  # Loop until all agents have terminated or truncated
    agent = env.agent_selection

    # Get observation for the current agent
    obs = env.observe(agent)
    # print(obs)

    # Select an action
    # Example: random legal action
    action_mask = info[agent]["action_mask"]
    #print(action_mask)
    legal_actions = [i for i, valid in enumerate(action_mask) if valid]
    #print(legal_actions)
    action = np.random.choice(legal_actions)
    # Step the environment
    env.step(action)
    env.render()

    # Get info for the current agent
    info = env.infos

    # Optional: print board for debugging
    print(f"Agent {agent} plays {action}")
    print(env.board)  # simplistic board view

# After loop ends, you can access cumulative rewards
print("Episode finished!")
for agent in ["X", "O"]:
    print(f"Agent {agent} cumulative reward: {env.rewards[agent]}") #env._cumulative_rewards[agent]
