import cv2
import math
import numpy as np
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv

class Odometry:
    '''
    Defines robot kinematics and inertial odometry calculations
    '''
    def __init__(
                self,
                initial_position = np.array((0,0,0))
                ):
       self.last_position = initial_position
       self.current_position = initial_position
       self.movement = initial_position

    def get_robot_movement(self, vx, vy, vw, elapsed_time):
        movement = np.array((vx, vy, vw))*elapsed_time
        return movement

    def update_robot_position(self, vx, vy, vw, elapsed_time):
        self.movement = self.get_robot_movement(vx, vy, vw, elapsed_time)
        self.current_position = self.last_position + self.movement
        self.last_position = self.current_position

if __name__ == "__main__":
    from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv

    env = SSLBaseEnv(
        field_type=1,
        n_robots_blue=0,
        n_robots_yellow=0,
        time_step=0.025)
    
    env.field.boundary_width = 0.3

    odometry = Odometry()
    
