import gym
import rsoccer_gym
import numpy as np

# Using VSS Single Agent env
env = gym.make('SSLPathPlanningObstacles-v0')
env = gym.wrappers.RecordEpisodeStatistics(env)
# env = gym.vector.make('SSLPathPlanning-v0', 2)
# import pdb; pdb.set_trace()
env.reset()
import pdb; pdb.set_trace()
# Run for 1 episode and print reward at the end
for i in range(1):
    done = False
    while True:
        # Step using random actions
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        env.render()
        if np.all(done):
            env.reset()
    print(info[0])
    print(reward)