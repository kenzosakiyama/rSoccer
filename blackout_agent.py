from time import sleep
import gym
import rsoccer_gym
import numpy as np
from rsoccer_gym.Tracking.ParticleFilterBase import ParticleFilter
from rsoccer_gym.Tracking import ResamplingAlgorithms

def set_robot_speed(env, vx, vy, vw):
    action = env.action_space.sample()
    action[0] = vx
    action[1] = vy
    action[2] = vw
    return action

def rotate_on_self(env, vw):
    action = set_robot_speed(env, 0, 0, vw)
    return action

def move_forward(env, vx):
    action = set_robot_speed(env, vx, 0, 0)
    return action

def move_random(env):
    return env.action_space.sample()

def split_observation(measurement):
    movement = measurement[:3]
    vision_xy_list = measurement[3:]

    vision_points = []
    for i in range(0, len(vision_xy_list)):
        if not (i%2):
            x, y = vision_xy_list[i], vision_xy_list[i+1]
            vision_points.append((x,y))
    
    return movement, np.array(vision_points)

if __name__ == "__main__":
    n_particles = 25
    vertical_lines_nr = 1

    seed_x, seed_y, seed_radius = 0, 0, 1

    # Using VSS Single Agent env
    env = gym.make('SSLVisionBlackout-v0', 
                vertical_lines_nr = vertical_lines_nr, 
                n_particles = n_particles)
    env.reset()

    robot_tracker = ParticleFilter(
                                    number_of_particles=n_particles, 
                                    field=env.field,
                                    process_noise=[1, 1, 1],
                                    measurement_noise=[1, 1],
                                    vertical_lines_nr=vertical_lines_nr,
                                    resampling_algorithm=ResamplingAlgorithms.STRATIFIED)
    # robot_tracker.initialize_particles_uniform()
    robot_tracker.initialize_particles_from_seed_position(seed_x, seed_y, seed_radius)

    # Run for 1 episode and print reward at the end
    for i in range(1):
        done = False
        while not done:
            action = move_random(env)
            measurements, _, _, _ = env.step(action)
            movement, vision_points = split_observation(measurements)
            robot_tracker.update(movement, vision_points)
            env.update_particles(robot_tracker.particles)
            env.render()
