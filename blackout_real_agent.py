import gym
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

def split_actions_list(vision_movements, env):
    actions = []
    for movement in vision_movements:
        vx, vy, vw = movement/env.time_step
        action = set_robot_speed(env, vx, vy, vw)
        actions.append(action)
    return actions

if __name__ == "__main__":
    import os
    from rsoccer_gym.Utils.load_localization_data import Read
    cwd = os.getcwd()

    n_particles = 50
    vertical_lines_nr = 1

    # LOAD REAL ODOMETRY DATA
    quadrado_nr = 15
    path = cwd+f'/localization_data/quadrado{quadrado_nr}/log.csv'
    data = Read(path)
    time_step_ms =  data.get_timesteps_average()
    time_steps = data.get_timesteps()

    # LOAD POSITION DATA
    position = data.get_position()

    # SET INITIAL ROBOT POSITION AND SEED
    initial_position = position[0]
    seed_radius = 1
    seed_x, seed_y, seed_theta = initial_position
    initial_position[2] = np.degrees(initial_position[2])

    # Using VSS Single Agent env
    env = gym.make('SSLVisionBlackout-v0', 
                vertical_lines_nr = vertical_lines_nr, 
                n_particles = n_particles,
                initial_position = initial_position,
                time_step=time_step_ms)
    env.reset()

    robot_tracker = ParticleFilter(
                                    number_of_particles=n_particles, 
                                    field=env.field,
                                    process_noise=[1, 1, 1],
                                    measurement_noise=[1, 1],
                                    vertical_lines_nr=vertical_lines_nr,
                                    resampling_algorithm=ResamplingAlgorithms.SYSTEMATIC)
    # robot_tracker.initialize_particles_uniform()
    robot_tracker.initialize_particles_from_seed_position(seed_x, seed_y, seed_radius)

    # movements list
    odometry = data.get_odometry()
    odometry_movements = data.get_odometry_movement(degrees=True)

    counter = 0

    while env.steps<len(position):
        env.update_time_step(time_steps[env.steps])
        robot_x, robot_y, robot_w = odometry[env.steps]
        action = position[env.steps]
        measurements, _, _, _ = env.step(action)
        _, vision_points = split_observation(measurements)
        dx, dy, dtheta = odometry_movements[env.steps]
        dx, dy = data.rotate_to_local(dx, dy, robot_w)
        movement = [dx, dy, dtheta]
        robot_tracker.update(movement, vision_points)
        odometry_tracking = [robot_x, robot_y, np.rad2deg(robot_w)]
        particles_filter_tracking = robot_tracker.get_average_state()         
        env.update_particles(robot_tracker.particles, odometry_tracking, particles_filter_tracking)
        env.render()
        if counter<0:
            env.update_step(0)
            counter += 1
