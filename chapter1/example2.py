import sys
sys.path.append("..")
from grid import GridWorld
import random
import numpy as np

# Example usage:
if __name__ == "__main__":             
    env = GridWorld()
    state = env.reset()
    state_values = np.zeros(env.num_states)

    policy_matrix=np.random.rand(env.num_states,len(env.action_space))
    for t in range(1000):
        for i in range(env.env_size[0]):
            for j in range(env.env_size[1]):
                state_index = i * env.env_size[0] + j
                action = random.choice(env.action_space)
                next_state, reward, done, info = env.step(action)
    