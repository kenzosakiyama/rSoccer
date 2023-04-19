from rsoccer_gym.Perception.ParticleVision import ParticleVision
from rsoccer_gym.Tracking.ParticleFilterBase import Particle
from rsoccer_gym.Perception.jetson_vision import JetsonVision
from rsoccer_gym.Utils.load_localization_data import Read
from rsoccer_gym.Perception.entities import Field

import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import os

def set_field_limits(field):
    field.x_min = -0.3
    field.x_max = 4.2
    field.y_min = -3
    field.y_max = 3
    return field

def get_image_from_frame_nr(path_to_images_folder, frame_nr):
    dir = path_to_images_folder+f'/cam/{frame_nr}.png'
    img = cv2.imread(dir)
    return img

def get_point_from_vision_process(robot_vision, img, has_goal, goal):
    if has_goal:
        print("FRAME HAS GOAL")
        cv2.rectangle(img, (int(goal[0]+1), int(goal[2]+1)), (int(goal[1]-1), int(goal[3]-1)), color=(0,255,0))

    _, _, tracked_goal, _, particle_filter_observations = robot_vision.process_from_log(src=img, 
                                                                                    timestamp=time.time(), 
                                                                                    has_goal=has_goal, 
                                                                                    goal_bounding_box=goal)


    goal = robot_vision.jetson_cam.xyToPolarCoordinates(tracked_goal.center_x, tracked_goal.center_y)
    boundary_ground_points, _ = particle_filter_observations
    points = []
    for point in boundary_ground_points:
        point = robot_vision.jetson_cam.xyToPolarCoordinates(point[0], point[1])
        points.append(point)
    return has_goal, goal, points

def compute_particle_observation(particle, particle_vision, field):
    goal = particle_vision.track_positive_goal_center(                                    
                                particle.x, 
                                particle.y, 
                                particle.theta, 
                                field)
    boundary_points = particle_vision.detect_boundary_points(
                                particle.x, 
                                particle.y, 
                                particle.theta, 
                                field)
    
    return goal, boundary_points

def serialize_observation_data_to_log(quadrado_nr, frame_nr, robot_boundary_points, particle_boundary_points):
    measured_distance = robot_boundary_points[0][0]
    angle = robot_boundary_points[0][0]
    expected_distance = particle_boundary_points[0][0]
    data = [quadrado_nr, frame_nr, expected_distance, measured_distance, angle]
    return data

if __name__ == "__main__":
    import csv

    cwd = os.getcwd()

    debug = True

    step = 1
    vertical_lines_nr=1
    robot_vision = JetsonVision(
                            vertical_lines_nr=vertical_lines_nr, 
                            enable_field_detection=True,
                            enable_randomized_observations=True,
                            debug=debug)
    robot_vision.jetson_cam.setPoseFrom3DModel(170, 107.2)

    particle_vision = ParticleVision(vertical_lines_nr=vertical_lines_nr)
    particle = Particle()
    field = Field()
    field.redefineFieldLimits(x_min=-0.3, 
                              x_max=4.2, 
                              y_min=-3, 
                              y_max=3)
    
    errors_log = []
    quadrados = [15] #TODO: remover falsos positivos das detecções de gol

    for quadrado_nr in quadrados:
        # LOAD DATASET FROM REAL EXPERIMENTS
        path = cwd+f'/localization_data/sqr_02'
        path_to_log = path+'/logs/processed.csv'
        data = Read(path_to_log, is_raw=False)
        frames = data.get_frames()
        has_goals = data.get_has_goals(remove_false_positives=True)
        goals = data.get_goals()

        # LOAD REAL POSITION DATA
        position = data.get_position()
        for i in range(0, len(frames), step):

            # MAKE ROBOT OBSERVATION
            img = get_image_from_frame_nr(path, frames[i])
            _, _, robot_boundary_points = get_point_from_vision_process(robot_vision,
                                                                        img,
                                                                        has_goals[i],
                                                                        goals[i])
            
            # MAKE PARTICLE OBSERVATION
            current_state = [position[i][0], position[i][1], np.rad2deg(position[i][2])]
            particle = Particle(weight=1, initial_state=current_state)
            if len(robot_boundary_points)>0:
                particle_vision.set_detection_angles_from_list([robot_boundary_points[0][1]])
                _, particle_boundary_points = compute_particle_observation(particle, particle_vision, field)
                data = serialize_observation_data_to_log(quadrado_nr, 
                                                        frames[i], 
                                                        robot_boundary_points,
                                                        particle_boundary_points)
                
                print(frames[i], robot_boundary_points, particle_boundary_points)        
                if debug:
                    cv2.imshow("BOUNDARY DETECTION", img)
                    key = cv2.waitKey(-1) & 0xFF
                    if key == ord('q'):
                        break
                    if key == ord('s'):
                        errors_log.append(data)
            
                else:
                    errors_log.append(data)

    dir = cwd+f"/observations_data/log.csv"
    fields = ["QUADRADO NR", "FRAME NR", "EXPECTED DIST", "MEASURED DIST", "ANGLE"]
    with open(dir, 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(errors_log)
                         
    
    
