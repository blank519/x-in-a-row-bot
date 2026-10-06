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

# Get observation for the current agent
obs = env._observe_for_learner()
print(obs)

# Step the environment a few times, choosing a random legal action each turn
max_steps = 10
for step_num in range(max_steps):
    if done:
        break

    # Select a random legal action from the current action mask
    action_mask = env.action_masks()
    legal_actions = np.flatnonzero(action_mask)
    action = int(np.random.choice(legal_actions))

    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    print(f"Step {step_num}: action={action}, reward={reward}, "
          f"terminated={terminated}, truncated={truncated}, info={info}")

print("Episode finished!" if done else "Reached max steps.")
