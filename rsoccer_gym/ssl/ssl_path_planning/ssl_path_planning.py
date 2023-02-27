import random
from rsoccer_gym.Render.Render import RCGymRender

from rsoccer_gym.ssl.ssl_path_planning.navigation import (
    Point2D,
    GoToPointEntry,
    go_to_point,
    abs_smallest_angle_diff,
    dist_to,
    length,
)

import gym
import numpy as np
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree


ANGLE_TOLERANCE: float = np.deg2rad(7.5)


class SSLPathPlanningEnv(SSLBaseEnv):
    """The SSL robot needs to reach the target point with a given angle"""

    def __init__(self, field_type=1, n_robots_yellow=0):
        super().__init__(
            field_type=field_type,
            n_robots_blue=1,
            n_robots_yellow=n_robots_yellow,
            time_step=0.025,
        )

        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32  # hyp tg.
        )

        n_obs = 4 + 7 * self.n_robots_blue + 2 * self.n_robots_yellow
        self.observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(n_obs,),
            dtype=np.float32,
        )

        # Limit robot speeds
        self.max_v = 2.5
        self.max_w = 10

        self.target_point: Point2D = Point2D(0, 0)
        self.target_speed: float = 0.0
        self.target_angle: float = 0.0
        self.cumulative_reward_info = {
            "reward_dist": 0,
            "reward_angle": 0,
            "reward_objective": 0,
            "Original_reward": 0,
        }

        print("Environment initialized")

    def reset(self):
        self.cumulative_reward_info = {
            "reward_dist": 0,
            "reward_angle": 0,
            "reward_objective": 0,
            "Original_reward": 0,
        }
        return super().reset()

    def step(self, action):
        observation, reward, done, _ = super().step(action)
        return observation, reward, done, self.cumulative_reward_info

    def _frame_to_observations(self):
        observation = list()

        observation.append(self.norm_pos(self.target_point.x))
        observation.append(self.norm_pos(self.target_point.y))
        observation.append(np.sin(self.target_angle))
        observation.append(np.cos(self.target_angle))
        # observation.append(self.norm_v(self.target_speed))

        # observation.append(self.norm_pos(self.frame.ball.x))
        # observation.append(self.norm_pos(self.frame.ball.y))
        # observation.append(self.norm_v(self.frame.ball.v_x))
        # observation.append(self.norm_v(self.frame.ball.v_y))

        for i in range(self.n_robots_blue):
            observation.append(self.norm_pos(self.frame.robots_blue[i].x))
            observation.append(self.norm_pos(self.frame.robots_blue[i].y))
            observation.append(np.sin(np.deg2rad(self.frame.robots_blue[i].theta)))
            observation.append(np.cos(np.deg2rad(self.frame.robots_blue[i].theta)))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_x))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_y))
            observation.append(self.norm_w(self.frame.robots_blue[i].v_theta))

        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))

        return np.array(observation, dtype=np.float32)

    def _get_commands(self, actions):
        commands = []

        angle = self.frame.robots_blue[0].theta
        v_x, v_y, v_theta = self.convert_actions(actions, np.deg2rad(angle))
        cmd = Robot(
            yellow=False,
            id=0,
            v_x=v_x,
            v_y=v_y,
            v_theta=v_theta,
            dribbler=False if actions[3] > 0 else False,
        )
        commands.append(cmd)

        return commands

    def convert_actions(self, action, angle):
        """Denormalize, clip to absolute max and convert to local"""
        # Denormalize
        v_x = action[0] * self.max_v
        v_y = action[1] * self.max_v
        v_theta = action[2] * self.max_w
        # Convert to local
        v_x, v_y = v_x * np.cos(angle) + v_y * np.sin(angle), -v_x * np.sin(
            angle
        ) + v_y * np.cos(angle)

        # clip by max absolute
        v_norm = np.linalg.norm([v_x, v_y])
        c = v_norm < self.max_v or self.max_v / v_norm
        v_x, v_y = v_x * c, v_y * c

        return v_x, v_y, v_theta

    def reward_function(
        self,
        robot_pos: Point2D,
        last_robot_pos: Point2D,
        robot_vel: Point2D,
        robot_angle: float,
        target_pos: Point2D,
        target_speed: float,
        target_angle: float,
    ):
        PESO_DIST = 0.5
        PESO_ANGULO = 0.5
        reward = np.zeros(2)
        done = False
        # max_dist = np.sqrt(self.field.length**2 + self.field.width**2)
        last_dist_robot_to_target = dist_to(target_pos, last_robot_pos)
        dist_robot_to_target = dist_to(target_pos, robot_pos)

        # robot_speed = length(robot_vel)

        dist_rw = (last_dist_robot_to_target - dist_robot_to_target) / self.max_v

        last_robot_angle = np.deg2rad(self.last_frame.robots_blue[0].theta)

        angle_dist_rw = (abs_smallest_angle_diff(last_robot_angle, target_angle) -
                         abs_smallest_angle_diff(robot_angle, target_angle)) / self.max_w
        angle_ok = abs_smallest_angle_diff(robot_angle, target_angle) < ANGLE_TOLERANCE

        if dist_robot_to_target < 0.2 and angle_ok:
            done = True
            objective_reward = 1
            reward += 0.5
            self.cumulative_reward_info["reward_objective"] += objective_reward

        reward[0] = dist_rw * PESO_DIST
        angle_ok = 1 if angle_ok else -1
        reward[1] = PESO_ANGULO * angle_dist_rw

        self.cumulative_reward_info["reward_dist"] += dist_rw
        self.cumulative_reward_info["reward_angle"] += angle_dist_rw
        self.cumulative_reward_info["Original_reward"] += (
            dist_rw * PESO_DIST + PESO_ANGULO * angle_dist_rw
        )
        return reward.sum(), done

    def _calculate_reward_and_done(self):
        robot = self.frame.robots_blue[0]
        last_robot = self.last_frame.robots_blue[0]

        robot_pos = Point2D(x=robot.x, y=robot.y)
        last_robot_pos = Point2D(x=last_robot.x, y=last_robot.y)
        robot_angle = np.deg2rad(robot.theta)
        target_pos = self.target_point
        target_speed = self.target_speed
        target_angle = self.target_angle

        robot_vel = Point2D(x=robot.v_x, y=robot.v_y)

        reward, done = self.reward_function(
            robot_pos=robot_pos,
            last_robot_pos=last_robot_pos,
            robot_vel=robot_vel,
            robot_angle=robot_angle,
            target_pos=target_pos,
            target_speed=target_speed,
            target_angle=target_angle,
        )
        return reward, done

    def _get_initial_positions_frame(self):
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def get_random_x():
            return random.uniform(-field_half_length + 0.1, field_half_length - 0.1)

        def get_random_y():
            return random.uniform(-field_half_width + 0.1, field_half_width - 0.1)

        def get_random_speed():
            return random.uniform(0, self.max_v)

        def get_random_theta():
            return random.uniform(0, 360)

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=get_random_x(), y=get_random_y())

        self.target_point = Point2D(x=get_random_x(), y=get_random_y())
        self.target_speed = get_random_speed()
        self.target_angle = np.deg2rad(get_random_theta())

        #  TODO: Move RCGymRender to another place
        self.view = RCGymRender(
            self.n_robots_blue,
            self.n_robots_yellow,
            self.field,
            simulator="ssl",
            angle_tolerance=ANGLE_TOLERANCE,
        )

        self.view.set_target(self.target_point.x, self.target_point.y)
        self.view.set_target_angle(np.rad2deg(self.target_angle))

        min_dist = 0.2

        places = KDTree()
        places.insert((self.target_point.x, self.target_point.y))
        places.insert((pos_frame.ball.x, pos_frame.ball.y))

        for i in range(self.n_robots_blue):
            pos = (get_random_x(), get_random_y())

            while places.get_nearest(pos)[1] < min_dist:
                pos = (get_random_x(), get_random_y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(
                id=i, yellow=False, x=pos[0], y=pos[1], theta=get_random_theta()
            )

        for i in range(self.n_robots_yellow):
            pos = (get_random_x(), get_random_y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (get_random_x(), get_random_y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(
                id=i, yellow=True, x=pos[0], y=pos[1], theta=get_random_theta()
            )

        return pos_frame
