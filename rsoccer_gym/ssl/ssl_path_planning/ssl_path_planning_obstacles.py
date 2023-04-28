import random
from rsoccer_gym.Render.Render import RCGymRender

from rsoccer_gym.ssl.ssl_path_planning.navigation import Point2D, GoToPointEntry, RobotMove, go_to_point, abs_smallest_angle_diff, dist_to, length

import gym
import numpy as np
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree

ANGLE_TOLERANCE: float = np.deg2rad(7.5) # 7.5 degrees
SPEED_TOLERANCE: float = 0.20 # m/s == 20 cm/s
DIST_TOLERANCE: float = 0.10 # m == 10 cm
ANGULAR_SPEED_TOLERANCE: float = 5 # 5 rad/s

class SSLPathPlanningObstaclesEnv(SSLBaseEnv):
    """The SSL robot needs to reach the target point with a given angle"""

    def __init__(self, field_type=1, n_robots=3):
        super().__init__(field_type=field_type, n_robots_blue=n_robots,
                         n_robots_yellow=0, time_step=0.025)
        self.n_robots = n_robots
        self.num_envs = n_robots
        self.is_vector_env = True
        self.max_episode_steps = 1200
        self.action_space = gym.spaces.Box(low=-1, high=1,  # hyp tg.
                                           shape=(n_robots, 3), dtype=np.float32)

        n_obs = 6 + 4 + 7 + 2*(n_robots - 1)
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(n_robots, n_obs),
                                                dtype=np.float32)

        # Limit robot speeds
        self.max_v = 2.5
        self.max_w = 10
        self.max_w_deg = np.rad2deg(self.max_w)

        self.target_point = [Point2D(0, 0)] * n_robots
        self.target_angle = [0.0] * n_robots
        self.target_velocity = [Point2D(0, 0)] * n_robots

        self.reward_info = [{
            'cumulative_dist_reward': 0,
            'cumulative_angle_reward': 0,
            'cumulative_velocity_reward': 0,
            'total_reward': 0,

            'dist_error': 0,
            'angle_error': 0,
            'velocity_error': 0,

            'current_speed': 0,
            'current_velocity_x': 0,
            'current_velocity_y': 0,
        }] * n_robots
        
        print('Environment initialized')
    
    def reset(self):
        self.reward_info = [{
            'cumulative_dist_reward': 0,
            'cumulative_angle_reward': 0,
            'cumulative_velocity_reward': 0,
            'total_reward': 0,

            'dist_error': 0,
            'angle_error': 0,
            'velocity_error': 0,

            'current_speed': 0,
            'current_velocity_x': 0,
            'current_velocity_y': 0,
        }] * self.n_robots
        return super().reset()
    
    def step(self, action):
        observation, reward, done, _ = super().step(action)
        self.last_observations = observation
        return observation, reward, done, self.reward_info

    def _frame_to_observations(self):
        observations = list()

        for i in range(self.n_robots):
            _obs = []
            _obs.append(self.norm_pos(self.target_point[i].x))
            _obs.append(self.norm_pos(self.target_point[i].y))
            _obs.append(np.sin(self.target_angle[i]))
            _obs.append(np.cos(self.target_angle[i]))
            _obs.append(self.norm_v(self.target_velocity[i].x))
            _obs.append(self.norm_v(self.target_velocity[i].y))

            _obs.append(self.norm_pos(self.frame.ball.x))
            _obs.append(self.norm_pos(self.frame.ball.y))
            _obs.append(self.norm_v(self.frame.ball.v_x))
            _obs.append(self.norm_v(self.frame.ball.v_y))

            _obs.append(self.norm_pos(self.frame.robots_blue[i].x))
            _obs.append(self.norm_pos(self.frame.robots_blue[i].y))
            _obs.append(np.sin(np.deg2rad(self.frame.robots_blue[i].theta)))
            _obs.append(np.cos(np.deg2rad(self.frame.robots_blue[i].theta)))
            _obs.append(self.norm_v(self.frame.robots_blue[i].v_x))
            _obs.append(self.norm_v(self.frame.robots_blue[i].v_y))
            _obs.append(self.norm_w(self.frame.robots_blue[i].v_theta))

            for j in range(self.n_robots):
                if i == j:
                    pass
                _obs.append(self.norm_pos(self.frame.robots_blue[i].x))
                _obs.append(self.norm_pos(self.frame.robots_blue[i].y))
            
            observations.append(_obs)

        return np.array(observations, dtype=np.float32)

    def convert_actions(self, move: RobotMove, angle):
        """Convert to local"""
        v_x = move.velocity.x
        v_y = move.velocity.y
        v_theta = move.angular_velocity

        # Convert to local
        v_x, v_y = v_x*np.cos(angle) + v_y*np.sin(angle),\
            -v_x*np.sin(angle) + v_y*np.cos(angle)

        # clip by max absolute
        v_norm = np.linalg.norm([v_x,v_y])
        c = v_norm < self.max_v or self.max_v / v_norm
        v_x, v_y = v_x*c, v_y*c
        
        return v_x, v_y, v_theta

    def _get_commands(self, action):
        commands = []

        for i in range(self.n_robots):
            target_v_x = action[i][0] * self.max_v
            target_v_y = action[i][1] * self.max_v
            target_v_w = action[i][2] * self.max_w

            move = RobotMove(
                velocity=Point2D(target_v_x, target_v_y),
                angular_velocity=target_v_w
            )

            v_x, v_y, v_theta = self.convert_actions(move, np.deg2rad(self.frame.robots_blue[0].theta))

            commands.append(Robot(yellow=False,id=i,v_x=v_x,v_y=v_y,v_theta=v_theta))
        return commands

    def reward_function(self, robot, last_robot, target_pos, target_angle, target_vel, info):
        max_dist = np.sqrt(self.field.length ** 2 + self.field.width ** 2)
        robot_pos = Point2D(robot.x, robot.y)
        robot_vel = Point2D(robot.v_x, robot.v_y)
        robot_angle = np.deg2rad(robot.theta)
        robot_angular_vel = np.deg2rad(robot.v_theta)
        last_robot_pos = Point2D(last_robot.x, last_robot.y)
        last_robot_vel = Point2D(last_robot.v_x, last_robot.v_y)
        last_robot_angle = np.deg2rad(last_robot.theta)

        last_dist_robot_to_target = dist_to(target_pos, last_robot_pos)
        dist_robot_to_target = dist_to(target_pos, robot_pos)

        last_angle_error = abs_smallest_angle_diff(last_robot_angle, target_angle)
        angle_error = abs_smallest_angle_diff(robot_angle, target_angle)

        last_robot_velocity_to_target = dist_to(target_vel, last_robot_vel)
        robot_velocity_to_target = dist_to(target_vel, robot_vel)

        angle_reward = 0.125 * (last_angle_error - angle_error) / np.pi
        dist_reward = 0.75 * (last_dist_robot_to_target - dist_robot_to_target) / max_dist
        velocity_reward = 0.125 * (last_robot_velocity_to_target - robot_velocity_to_target) / self.max_v
        
        angular_velocity_to_target = abs(robot_angular_vel)

        info['dist_error'] = dist_robot_to_target
        info['angle_error'] = angle_error
        info['velocity_error'] = robot_velocity_to_target
        info['angular_velocity'] = angular_velocity_to_target

        info['current_speed'] = length(robot_vel)
        info['current_velocity_x'] = robot_vel.x
        info['current_velocity_y'] = robot_vel.y

        info['total_reward'] += dist_reward
        info['cumulative_dist_reward'] += dist_reward

        info['total_reward'] += angle_reward
        info['cumulative_angle_reward'] += angle_reward

        if dist_robot_to_target <= DIST_TOLERANCE:
            info['total_reward'] += velocity_reward
            info['cumulative_velocity_reward'] += velocity_reward

            if robot_velocity_to_target <= SPEED_TOLERANCE and angular_velocity_to_target <= ANGULAR_SPEED_TOLERANCE:
                if angle_error <= ANGLE_TOLERANCE:
                    return angle_reward + 3, True
                else:
                    return angle_reward, False

            return angle_reward + velocity_reward, False

        return dist_reward + angle_reward, False

    def _calculate_reward_and_done(self):
        rewards, dones = [], []
        for i in range(self.n_robots):
            robot = self.frame.robots_blue[i]
            last_robot = self.last_frame.robots_blue[i]
            target_pos = self.target_point[i]
            target_angle = self.target_angle[i]
            target_vel = self.target_velocity[i]
            _rew, _done = self.reward_function(robot, last_robot, target_pos, target_angle, target_vel, self.reward_info[i])
            # TODO: IF DONE, RESET TARGET AND INFOS
            if self.steps >= self.max_episode_steps:
                self.reward_info[i]["TimeLimit.truncated"] = not _done
                _done = True
            if _done:
                self.reward_info[i]["terminal_observation"] = self.last_observations[i].copy()
            rewards.append(_rew)
            dones.append(_done)
        return rewards, dones

    def _get_initial_positions_frame(self):
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def get_random_x():
            return random.uniform(-field_half_length + 0.1,
                                  field_half_length - 0.1)

        def get_random_y():
            return random.uniform(-field_half_width + 0.1,
                                  field_half_width - 0.1)

        def get_random_theta():
            return random.uniform(0, 360)

        pos_frame: Frame = Frame()
        pos_frame.ball = Ball(x=0, y=20)

        min_gen_dist = 0.2

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))

        for i in range(self.n_robots):
            pos = (get_random_x(), get_random_y())

            while places.get_nearest(pos)[1] < min_gen_dist:
                pos = (get_random_x(), get_random_y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(id=i, yellow=False,x=pos[0], y=pos[1], theta=get_random_theta())

            pos = (get_random_x(), get_random_y())
            while places.get_nearest(pos)[1] < min_gen_dist:
                pos = (get_random_x(), get_random_y())
            self.target_point[i] = Point2D(pos[0], pos[1])
            self.target_angle[i] = np.deg2rad(get_random_theta())
            self.target_velocity[i] = Point2D(0, 0)

        return pos_frame

    def render(self, mode = 'human') -> None:
        '''
        Renders the game depending on 
        ball's and players' positions.

        Parameters
        ----------
        None

        Returns
        -------
        None

        '''
        if self.view == None:
            self.view = RCGymRender(self.n_robots_blue,
                                    self.n_robots_yellow,
                                    self.field,
                                    simulator='ssl',
                                    angle_tolerance=ANGLE_TOLERANCE)

        # TODO: render for all robots
        # self.view.set_target(self.target_point[0].x, self.target_point[0].y)
        # self.view.set_target_angle(np.rad2deg(self.target_angle[0]))

        return self.view.render_frame(self.frame, return_rgb_array=mode == "rgb_array", target_points=self.target_point, target_angles=self.target_angle)