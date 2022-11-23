import random

import gym
import numpy as np
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree
from rsoccer_gym.Perception.Vision import Camera, SSLEmbeddedVision
from rsoccer_gym.Perception.Odometry import Odometry
from rsoccer_gym.Tracking.ParticleFilterBase import Particle

class SSLVisionBlackoutEnv(SSLBaseEnv):
    """
        The SSL robot needs localize itself inside the field using Adaptive Monte Localization

        Description:
            One blue robot is randomly placed on a div B field,
            it has a seed of its initial position and
            the episode ends when the robots position confidence reaches ...% (how much?)

        Observation:
            Type: Box(3 + 2*vertical_lines_nr)
            Num      Observation normalized  
            0->2     Robot Odometry         [X, Y, W]
            3+       Field Boundary Points  [X, Y]

        Actions:
            Type: Box(3, )
            Num     Action
            0       id 0 Blue Global X Direction Speed  (%)
            1       id 0 Blue Global Y Direction Speed  (%)
            2       id 0 Blue Angular Speed  (%)

        Reward:
            1 if pose confidence is higher than threshold

        Starting State:
            Randomized robot initial position

        Episode Termination:
            Pose confidence is higher than threshold or 30 seconds (1200 steps)
    """

    def __init__(self, initial_position=[], field_type=1, vertical_lines_nr=1, n_particles=0):
        super().__init__(field_type=field_type, 
                        n_robots_blue=1, 
                        n_robots_yellow=0, 
                        n_particles=n_particles,
                        time_step=0.005)
        
        self.field.boundary_width = 0.3
        self.embedded_vision = SSLEmbeddedVision(vertical_lines_nr=vertical_lines_nr)

        self.odometry = Odometry()
        self.particles = {}

        # LOADS VISION POSITION DATA
        if len(initial_position)>0:
            self.initial_position =  initial_position
            self.using_log_data = True
        else:
            self.using_log_data = False


        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(3, ), dtype=np.float32)
        
        n_obs = 3 + 2*vertical_lines_nr
        self.observation_space = gym.spaces.Box(low=-(self.field.length/2+self.field.boundary_width),
                                                high=(self.field.length/2+self.field.boundary_width),
                                                shape=(n_obs, ),
                                                dtype=np.float32)
        
        # Limit robot speeds
        self.max_v = 2.5
        self.max_w = 10

        print('Environment initialized')

    def update_particles(self, particles):
        self.particles = particles

    def _render_particles(self):
        for i in range(self.n_particles):
            self.frame.particles[i] = self.particles[i]

    def _frame_to_observations(self):

        observation = []

        for i in range(self.n_robots_blue):
            movement = self.odometry.get_robot_movement(
                                self.frame.robots_blue[i].v_x,
                                self.frame.robots_blue[i].v_y,
                                self.frame.robots_blue[i].v_theta,
                                self.time_step)
            observation.append(movement[0])
            observation.append(movement[1])
            observation.append(movement[2])
            
            boundary_points = self.embedded_vision.detect_boundary_points(
                                self.frame.robots_blue[i].x, 
                                self.frame.robots_blue[i].y,
                                self.frame.robots_blue[i].theta, 
                                self.field)
            for point in boundary_points:
                observation.append(point[0])
                observation.append(point[1])

        return np.array(observation, dtype=np.float32)

    def _get_commands(self, actions):
        commands = []
        angle = self.frame.robots_blue[0].theta
        v_x, v_y, v_theta = self.convert_actions(actions, np.deg2rad(angle))
        cmd = Robot(yellow=False, id=0, v_x=v_x, v_y=v_y, v_theta=v_theta)
        commands.append(cmd)
        return commands

    def convert_actions(self, action, angle):
        """Denormalize, clip to absolute max and convert to local"""
        # Denormalize
        v_x = action[0] * self.max_v
        v_y = action[1] * self.max_v
        v_theta = action[2] * self.max_w
        # Convert to local
        v_x, v_y = v_x*np.cos(angle) + v_y*np.sin(angle),\
            -v_x*np.sin(angle) + v_y*np.cos(angle)

        # clip by max absolute
        v_norm = np.linalg.norm([v_x,v_y])
        c = v_norm < self.max_v or self.max_v / v_norm
        v_x, v_y = v_x*c, v_y*c
        
        return v_x, v_y, v_theta

    def _calculate_reward_and_done(self):
        reward = 0

        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        
        dist_robot_ball = np.linalg.norm(
            np.array([ball.x, ball.y]) 
            - np.array([robot.x, robot.y])
        )
        
        # Check if robot is less than 0.2m from ball
        if dist_robot_ball < 0.2:
            reward = 1

        done = reward

        return reward, done
    
    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def x(): return random.uniform(-field_half_length + 0.1,
                                       field_half_length - 0.1)

        def y(): return random.uniform(-field_half_width + 0.1,
                                       field_half_width - 0.1)

        def theta(): return random.uniform(0, 360)

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=5, y=0)

        min_dist = 0.2

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))

        if self.using_log_data:
            for i in range(self.n_robots_blue):
                pos = self.initial_position
                # import pdb;pdb.set_trace()
                pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=pos[2])
                initial_position = np.array((
                        pos_frame.robots_blue[i].x, 
                        pos_frame.robots_blue[i].y, 
                        pos_frame.robots_blue[i].theta))
                self.odometry.__init__(initial_position)            

        else:
            for i in range(self.n_robots_blue):
                pos = (x(), y())
                while places.get_nearest(pos)[1] < min_dist:
                    pos = (x(), y())

                places.insert(pos)
                pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=theta())
                initial_position = np.array((
                        pos_frame.robots_blue[i].x, 
                        pos_frame.robots_blue[i].y, 
                        pos_frame.robots_blue[i].theta))
                self.odometry.__init__(initial_position)
        
        for i in range(self.n_particles):
            pos_frame.particles[i] = Particle()

        return pos_frame
