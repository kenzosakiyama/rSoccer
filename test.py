import gym
import rsoccer_gym
import numpy as np

# Using VSS Single Agent env
env = gym.make('SSLPathPlanning-v0')
# env = gym.vector.make('SSLPathPlanning-v0', 2)
import pdb; pdb.set_trace()
env.reset()
# Run for 1 episode and print reward at the end
for i in range(1):
    done = False
    while not np.any(done):
        # Step using random actions
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        # env.render()
    print(reward)