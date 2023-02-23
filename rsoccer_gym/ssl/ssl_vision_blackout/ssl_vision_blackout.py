import random

import gym
import numpy as np
import time
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree
from rsoccer_gym.Perception.ParticleVision import Camera, SSLEmbeddedVision
from rsoccer_gym.Perception.Odometry import Odometry
from rsoccer_gym.Tracking.ParticleFilterBase import Particle
from rsoccer_gym.Perception.jetson_vision import JetsonVision


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

    def __init__(self, initial_position=[], time_step=0.005, field_type=1, vertical_lines_nr=1, n_particles=0, using_vision_frames=False):
        super().__init__(field_type=field_type, 
                        n_robots_blue=1, 
                        n_robots_yellow=0, 
                        n_particles=n_particles,
                        time_step=time_step)
        
        self.field.boundary_width = 0.3
        
        self.odometry = Odometry()
        self.particles = {}
        self.trackers = {}

        # LOADS VISION POSITION DATA
        self.using_vision_frames = False
        if len(initial_position)>0:
            self.initial_position =  initial_position
            self.using_log_data = True
            self.using_vision_frames = using_vision_frames
        else:
            self.using_log_data = False

        if self.using_vision_frames:
            self.embedded_vision = JetsonVision(
                                    vertical_lines_nr=vertical_lines_nr, 
                                    enable_field_detection=True,
                                    enable_randomized_observations=True)
            self.embedded_vision.jetson_cam.setPoseFrom3DModel(170, 106.7)
            self.img = np.zeros((480, 640, 3), dtype=np.uint8)
            self.has_goal = False
            self.goal_bbox = [0, 0, 0, 0]
        else:
            self.embedded_vision = SSLEmbeddedVision(vertical_lines_nr=vertical_lines_nr)

        self.set_field_limits(self.field, self.using_vision_frames)

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

    def set_field_limits(self, field, using_real_field = False):
        if using_real_field:
            self.field.x_min = -0.3
            self.field.x_max = 4.2
            self.field.y_min = -3
            self.field.y_max = 3
        else:
            self.field.x_min = -(field.length/2+field.boundary_width)
            self.field.x_max = (field.length/2+field.boundary_width)
            self.field.y_min = -(field.width/2+field.boundary_width)
            self.field.y_max = (field.width/2+field.boundary_width)        

    def update_particles(self, particles, odometry_tracking, particle_filter_tracking):
        self.particles = particles
        self.trackers[0] = Particle(odometry_tracking, 0.2)
        self.trackers[1] = Particle(particle_filter_tracking, 0.2)

    def update_step(self, step):
        self.steps = step
    
    def update_time_step(self, time_step):
        self.time_step = time_step

    def update_img(self, img, has_goal, goal_bbox):
        self.img = img
        self.has_goal = has_goal
        self.goal_bbox = goal_bbox

    def _render_particles(self):
        for i in range(self.n_particles):
            self.frame.particles[i] = self.particles[i]
        self.frame.trackers = self.trackers

    def _frame_to_observations(self):

        observation = []
        
        if self.using_vision_frames:
            _, _, tracked_goal, _, particle_filter_observations = self.embedded_vision.process_from_log(src=self.img, 
                                                                                            timestamp=time.time(), 
                                                                                            has_goal=self.has_goal, 
                                                                                            goal_bounding_box=self.goal_bbox)
            goal = self.embedded_vision.jetson_cam.xyToPolarCoordinates(tracked_goal.center_x, tracked_goal.center_y)
            observation.append(self.has_goal)
            observation.append(goal[0])
            observation.append(goal[1])

            boundary_ground_points, line_ground_points = particle_filter_observations
            for point in boundary_ground_points:
                point = self.embedded_vision.jetson_cam.xyToPolarCoordinates(point[0], point[1])
                observation.append(point[0])
                observation.append(point[1])
        else:
            boundary_points = self.embedded_vision.detect_boundary_points_random(
                                self.frame.robots_blue[0].x, 
                                self.frame.robots_blue[0].y,
                                self.frame.robots_blue[0].theta, 
                                self.field)
            for point in boundary_points:
                observation.append(point[0])
                observation.append(point[1])

        return np.array(observation, dtype=np.float32)

    def _get_commands(self, actions):
        commands = []
        if self.using_log_data:
            x, y, theta = actions[0], actions[1], np.rad2deg(actions[2])
            robot = Robot(yellow=False, id=0, x=x, y=y, theta=theta, v_x=0, v_y=0, v_theta=0)
            self.frame.robots_blue[0] = robot
            self.rsim.reset(self.frame)
            cmd = robot
        else:
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
