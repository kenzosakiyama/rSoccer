import gym
import rsoccer_gym

# env = gym.make('VSS-v0')
env = gym.make('VSSFIRA-v0', fira_port=10010)

env.reset()
done = False
import time
a = time.time()
step = 0
while not done:
    _, _, done, _ = env.step([1, 1])
    step += 1

print(step/(time.time()-a))