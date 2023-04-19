import gym
import numpy as np
import cv2
from rsoccer_gym.Tracking.ParticleFilterBase import ParticleFilter, Particle
from rsoccer_gym.Tracking import ResamplingAlgorithms
from rsoccer_gym.Plotter.Plotter import RealTimePlotter

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
    dir = path_to_images_folder+f'/cam/{frame_nr}.png'
    img = cv2.imread(dir)
    return img

def parse_particle_filter_data(measurements, global_movement, robot_w, data):
    goal, field_points = split_observation(measurements)
    dx, dy, dtheta = global_movement
    dx, dy = data.rotate_to_local(dx, dy, robot_w)
    movement = [dx, dy, dtheta]
    return movement, goal, field_points

if __name__ == "__main__":
    import os
    from rsoccer_gym.Utils.load_localization_data import Read
    import time
    cwd = os.getcwd()

    n_particles = 100
    vertical_lines_nr = 1

    # LOAD REAL ODOMETRY DATA
    # CHOOSE SCENARIO
    scenario = 'sqr'
    lap = 2
    path = f'/home/rc-blackout/ssl-navigation-dataset/data/{scenario}_0{lap}'
    path_to_log = path+'/logs/processed.csv'
    data = Read(path_to_log, is_raw=False)
    time_step_ms =  data.get_timesteps_average()
    time_steps = data.get_timesteps()
    frames = data.get_frames()
    has_goals = data.get_has_goals(remove_false_positives=True)
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

    robot_tracker = ParticleFilter(number_of_particles=n_particles, 
                                   field=env.field,
                                   process_noise=[1, 1, 0.1],
                                   measurement_noise=[1, 1],
                                   vertical_lines_nr=vertical_lines_nr,
                                   using_real_field=env.using_vision_frames,
                                   resampling_algorithm=ResamplingAlgorithms.SYSTEMATIC)
    robot_tracker.initialize_particles_from_seed_position(seed_x, seed_y, seed_radius)
    #robot_tracker.initialize_particles_uniform()

    # movements list
    odometry = data.get_odometry()
    odometry_movements = data.get_odometry_movement(degrees=True)
    odometry_particle = Particle(initial_state=initial_position,
                                 movement_deviation=[0, 0, 0])

    counter = 0

    #ax_titles = ["RMSE",
    #             "Similarity from Particles Filter Median Particle",
    #             "Non-normalized Weights Sum"]
    #plotter = RealTimePlotter(n_plots=3, 
    #                          max_size=len(position),
    #                          titles=ax_titles)

    timestamps = data.get_timestamps()[:-2]
    ground_truth_trajectory = data.get_position()[:-2]
    odometry_trajectory = []
    mcl_trajectory = []

    while env.steps<len(position)-1:
        img = get_image_from_frame_nr(path, frames[env.steps])
        env.update_img(img, has_goals[env.steps], goals[env.steps])
        action = position[env.steps]
        measurements, _, _, _ = env.step(action)
        movement, goal, field_points = parse_particle_filter_data(measurements,
                                                                  odometry_movements[env.steps],
                                                                  odometry[env.steps][2],
                                                                  data)
        robot_tracker.update(movement, goal, field_points)
        odometry_particle.move(movement)
        particles_filter_tracking = robot_tracker.get_average_state()         
        env.update_particles(robot_tracker.particles, 
                             odometry_particle.state,
                             particles_filter_tracking, 
                             time_steps[env.steps])
        
        # SAVE TRACKING DATA
        odometry_trajectory.append([odometry_particle.state[0], odometry_particle.state[1], np.deg2rad(odometry_particle.state[2])])
        mcl_trajectory.append([particles_filter_tracking[0], particles_filter_tracking[1], np.deg2rad(particles_filter_tracking[2])])

        #RMSE = np.sqrt((particles_filter_tracking[0]-position[env.steps][0])**2 + \
        #                (particles_filter_tracking[1]-position[env.steps][1])**2)
        #covariance = robot_tracker.compute_covariance(particles_filter_tracking)
        env.render()
        if counter<0:
            env.update_step(0)
            counter += 1

        #_ys = [RMSE, robot_tracker.average_particle_weight, covariance]
        #plotter.add_data(env.steps, _ys)
#
#    plotter.kill_process()
    
    np.savetxt(path+'/logs/ground_truth_trajectory.txt', ground_truth_trajectory)
    np.savetxt(path+'/logs/odometry_trajectory.txt', odometry_trajectory)
    np.savetxt(path+'/logs/mcl_trajectory.txt', mcl_trajectory)
    np.savetxt(path+'/logs/timestamps.txt', timestamps)

    print("finished")