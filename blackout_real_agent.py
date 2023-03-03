import gym
import numpy as np
import cv2
from rsoccer_gym.Tracking.ParticleFilterBase import ParticleFilter
from rsoccer_gym.Tracking import ResamplingAlgorithms
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import multiprocessing as mp

def split_observation(measurements):
    goal = measurements[:3]

    field_points = measurements[3:]
    vision_points = []
    for i in range(0, len(field_points)):
        if not (i%2):
            x, y = field_points[i], field_points[i+1]
            vision_points.append((x,y))
    return goal, np.array(vision_points)

def get_image_from_frame_nr(path_to_images_folder, frame_nr):
    dir = path_to_images_folder+f'/{frame_nr}.jpg'
    img = cv2.imread(dir)
    return img

def parse_particle_filter_data(measurements, global_movement, robot_w, data):
    goal, field_points = split_observation(measurements)
    dx, dy, dtheta = global_movement
    dx, dy = data.rotate_to_local(dx, dy, robot_w)
    movement = [dx, dy, dtheta]
    return movement, goal, field_points

def process_plot(xs, ys):
    n_plots = 3
    fig = plt.figure(figsize=(6.75,9.3))
    ax = fig.subplots(nrows=n_plots, ncols=1)
    fig.subplots_adjust(left=0.1, 
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    list_x = []
    list_ys = []
    ax_titles = ["RMSE",
                 "Similarity from Particles Filter Median Particle",
                 "Non-normalized Weights Sum"]
    
    for i in range(0, n_plots):
        list_ys.append([])

    def update(*args, **kwargs):
        list_x = args[1]
        list_ys = args[2]
        _x = xs.get()
        _y = ys.get()
        list_x.append(_x)
        for i in range(0, n_plots):
            list_ys[i].append(_y[i])
            ax[i].clear()
            ax[i].plot(list_x, list_ys[i])
            ax[i].set_title(ax_titles[i])

        # Draw x and y lists
#        list_x = list_x[-20:]
#        list_y = list_y[-20:]

    _ = animation.FuncAnimation(fig, update, fargs=(list_x, list_ys,), interval=1)
    plt.show()

if __name__ == "__main__":
    import os
    from rsoccer_gym.Utils.load_localization_data import Read
    import time
    cwd = os.getcwd()

    n_particles = 50
    vertical_lines_nr = 1

    # LOAD REAL ODOMETRY DATA
    quadrado_nr = 15
    path = cwd+f'/localization_data/quadrado{quadrado_nr}'
    path_to_log = path+'/log.csv'
    data = Read(path_to_log)
    time_step_ms =  data.get_timesteps_average()
    time_steps = data.get_timesteps()
    frames = data.get_frames()
    has_goals = data.get_has_goals()
    goals = data.get_goals()

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
                time_step=time_step_ms,
                using_vision_frames = True)
    env.reset()

    robot_tracker = ParticleFilter(
                                    number_of_particles=n_particles, 
                                    field=env.field,
                                    process_noise=[1, 1, 1],
                                    measurement_noise=[1, 1],
                                    vertical_lines_nr=vertical_lines_nr,
                                    using_real_field=env.using_vision_frames,
                                    resampling_algorithm=ResamplingAlgorithms.SYSTEMATIC)
    robot_tracker.initialize_particles_from_seed_position(seed_x, seed_y, seed_radius)
    #robot_tracker.initialize_particles_uniform()

    # movements list
    odometry = data.get_odometry()
    odometry_movements = data.get_odometry_movement(degrees=True)

    counter = 0

    xs = mp.Queue(maxsize=len(position))
    ys = mp.Queue(maxsize=3*len(position))

    side_process = mp.Process(target=process_plot, args=(xs, ys))
    side_process.start()

    while env.steps<len(position)-1:
        img = get_image_from_frame_nr(path, frames[env.steps])
        env.update_img(img, has_goals[env.steps], goals[env.steps])
        robot_x, robot_y, robot_w = odometry[env.steps]
        action = position[env.steps]
        measurements, _, _, _ = env.step(action)
        movement, goal, field_points = parse_particle_filter_data(
                                                                measurements, 
                                                                odometry_movements[env.steps],
                                                                robot_w,
                                                                data)
        robot_tracker.update(movement, goal, field_points)
        odometry_tracking = [robot_x, robot_y, np.rad2deg(robot_w)]
        particles_filter_tracking = robot_tracker.get_average_state()         
        env.update_particles(robot_tracker.particles, odometry_tracking, particles_filter_tracking, time_steps[env.steps])
        RMSE = np.sqrt((particles_filter_tracking[0]-position[env.steps][0])**2 + (particles_filter_tracking[1]-position[env.steps][1])**2)
        env.render()
        if counter<0:
            env.update_step(0)
            counter += 1

        _ys = [RMSE, robot_tracker.average_particle_weight, robot_tracker.prior_sum_weights/robot_tracker.n_particles]
        xs.put(env.steps)
        ys.put(_ys)

    side_process.terminate()
    side_process.join()