import gym
import numpy as np
import time
from rsoccer_gym.Tracking.ParticleFilterBase import ParticleFilter
from rsoccer_gym.Tracking import ResamplingAlgorithms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import multiprocessing as mp


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
        if not (i % 2):
            x, y = vision_xy_list[i], vision_xy_list[i + 1]
            vision_points.append((x, y))

    return movement, np.array(vision_points)


def split_actions_list(vision_movements, env):
    actions = []
    for movement in vision_movements:
        vx, vy, vw = movement / env.time_step
        action = set_robot_speed(env, vx, vy, vw)
        actions.append(action)
    return actions


def balaka(xs, ys):
    fig = plt.figure()
    ax = fig.add_subplot()
    def balakinha(*args, **kwargs):
        list_x = args[1]
        list_y = args[2]
        x = xs.get()
        y = ys.get()
        list_x.append(x)
        list_y.append(y)
        # Draw x and y lists
        ax.clear()
        ax.plot(list_x, list_y)
        list_x = list_x[-20:]
        list_y = list_y[-20:]

    big_x = []
    big_y = []
    _ = animation.FuncAnimation(fig, balakinha, fargs=(big_x, big_y),interval=33)
    plt.show()


if __name__ == "__main__":
    import os
    from rsoccer_gym.Utils.load_odometry_data import Read

    cwd = os.getcwd()

    n_particles = 50
    vertical_lines_nr = 1

    # LOAD REAL ODOMETRY DATA
    quadrado_nr = 15
    path = cwd + f"/odometry_data/quadrado_{quadrado_nr}.csv"
    data = Read(path)

    # LOAD POSITION DATA
    vision = data.get_vision()

    # SET INITIAL ROBOT POSITION AND SEED
    initial_position = vision[0]
    seed_radius = 1
    seed_x, seed_y, seed_theta = initial_position
    initial_position[2] = np.degrees(initial_position[2])

    # Using VSS Single Agent env
    env = gym.make('SSLVisionBlackout-v0', 
                vertical_lines_nr = vertical_lines_nr, 
                n_particles = n_particles,
                initial_position = initial_position,
                time_step=0.033,
                using_vision_frames = False)
    env.reset()

    robot_tracker = ParticleFilter(
                                    number_of_particles=n_particles, 
                                    field=env.field,
                                    process_noise=[1, 1, 1],
                                    measurement_noise=[1, 1],
                                    vertical_lines_nr=vertical_lines_nr,
                                    using_real_field = env.using_vision_frames,
                                    resampling_algorithm=ResamplingAlgorithms.SYSTEMATIC)
    # robot_tracker.initialize_particles_uniform()
    robot_tracker.initialize_particles_from_seed_position(seed_x, seed_y, seed_radius)
    # movements list
    odometry = data.get_odometry()
    vision_movements = data.get_vision_movement(degrees=True)
    odometry_movements = data.get_odometry_movement(degrees=True)

    counter = 0

    xs = mp.Queue(maxsize=len(vision))
    ys = mp.Queue(maxsize=len(vision))
    side_process = mp.Process(target=balaka, args=(xs, ys))
    side_process.start()

    while env.steps < len(vision):
        robot_x, robot_y, robot_w = odometry[env.steps]
        action = vision[env.steps]
        measurements, _, _, _ = env.step(action)
        _, vision_points = split_observation(measurements)
        dx, dy, dtheta = odometry_movements[env.steps]
        dx, dy = data.rotate_to_local(dx, dy, robot_w)
        movement = [dx, dy, dtheta]
        robot_tracker.update(movement, vision_points)
        odometry_tracking = [robot_x, robot_y, np.rad2deg(robot_w)]
        particles_filter_tracking = robot_tracker.get_average_state()
        env.update_particles(
            robot_tracker.particles, odometry_tracking, particles_filter_tracking
        )
        env.render()
        time.sleep(env.time_step)

        xs.put(env.steps)
        ys.put(robot_w)

    side_process.terminate()
    side_process.join()
