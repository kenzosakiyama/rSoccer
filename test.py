import gym
import rsoccer_gym
import numpy as np

# Using VSS Single Agent env
env = gym.make('VSSact-v0')

env.reset()

# Run for 1 episode and print reward at the end
for i in range(1):
    done = False
    j = 0
    while not done:
        # Step using random actions
        # action = env.action_space.sample()
        for k in range(3):
            obs_list = list()
            while j < 301:
                action = (0.5, 0.5)
                if j < 20:
                    next_state, velocities, done, _ = env.step(action)

                elif j > 20 and j < 30:
                    action = (-0.28, 0.28)
                    next_state, velocities, done, _ = env.step(action)

                elif j > 31 and j < 80:
                    action = (0.2, 0.2)
                    next_state, velocities, done, _ = env.step(action)

                elif j > 91 and j < 100:
                    action = (-0.2599999999999999999999,
                              0.24999999999999999999999)
                    next_state, velocities, done, _ = env.step(action)
                elif j > 101 and j < 191:
                    action = (0.2, 0.2)
                    next_state, velocities, done, _ = env.step(action)
                elif j > 191 and j < 200:
                    action = (-0.2599999999999999999999,
                              0.24999999999999999999999)
                    next_state, velocities, done, _ = env.step(action)
                elif j > 191 and j < 250:
                    action = (0.2, 0.2)
                    next_state, velocities, done, _ = env.step(action)
                elif j > 250 and j < 259:
                    action = (-0.2599999999999999999999,
                              0.24999999999999999999999)
                    next_state, velocities, done, _ = env.step(action)
                elif j > 265 and j < 299:
                    action = (0.2, 0.2)
                    next_state, velocities, done, _ = env.step(action)

                elif j > 299:
                    print('parei')
                    print(next_state)
                    break
                obs_list.append(np.concatenate((next_state,
                                                np.array(velocities)), axis=0))
                # print(obs_list[0].shape)
                j += 1
                env.render()
            obs_arr = np.array(obs_list)
            np.savetxt('./data/positions-' +
                       str(k) + '.txt', obs_arr, delimiter=',')
            print(k)
            j = 0
        break
    # print(reward)
