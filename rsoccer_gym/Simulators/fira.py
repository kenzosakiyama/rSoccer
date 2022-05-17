import socket
from typing import Dict, List

import numpy as np
from rsoccer_gym.Entities import Robot, Field, FrameVSSPB
from rsoccer_gym.Simulators.rsim import RSim

import rsoccer_gym.Simulators.pb_fira.packet_pb2 as packet_pb2
from rsoccer_gym.Simulators.pb_fira.state_pb2 import *
import subprocess
import os
import time

class Fira(RSim):
    def __init__(
        self,
        vision_ip="127.0.0.1",
        vision_port=10010,
        cmd_ip="127.0.0.1",
        cmd_port=20011,
    ):
        """
        Init SSLClient object.
        Extended description of function.
        Parameters
        ----------
        ip : str
            Multicast IP in format '255.255.255.255'.
        port : int
            Port up to 1024.
        """
        bin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin', 'FIRASim')
        cmd = [bin_path, '-H', '-vp', f'{vision_port}']
        self.p_fira = subprocess.Popen(cmd)

        self.vision_ip = vision_ip
        self.vision_port = vision_port
        self.com_ip = cmd_ip
        self.com_port = cmd_port
        self.com_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.com_address = (self.com_ip, self.com_port)

        self.vision_sock = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )
        self.vision_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.vision_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 128)
        self.vision_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        self.vision_sock.bind((self.vision_ip, self.vision_port))
        time.sleep(4) # TODO: sleep here is to wait for fira to initialize
        self.send_commands([])
        self.get_frame()

    def get_field_params(self):
        return Field(
            length=1.5,
            width=1.3,
            penalty_length=0.15,
            penalty_width=0.7,
            goal_width=0.4,
            goal_depth=0.1,
            ball_radius=0.0215,
            rbt_distance_center_kicker=-1.0,
            rbt_kicker_thickness=-1.0,
            rbt_kicker_width=-1.0,
            rbt_wheel0_angle=90.0,
            rbt_wheel1_angle=270.0,
            rbt_wheel2_angle=-1.0,
            rbt_wheel3_angle=-1.0,
            rbt_radius=0.0375,
            rbt_wheel_radius=0.02,
            rbt_motor_max_rpm=1000.0,
        )

    def __del__(self):
        self.p_fira.terminate()

    def reset(self, frame: FrameVSSPB):
        placement_pos = self._placement_dict_from_frame(frame)
        pkt = packet_pb2.Packet()

        ball_pos = placement_pos["ball_pos"][:2]
        ball_pkt = pkt.replace.ball
        ball_pkt.x = ball_pos[0]
        ball_pkt.y = ball_pos[1]

        robots_pkt = pkt.replace.robots
        for i, robot in enumerate(placement_pos["blue_robots_pos"]):
            rep_rob = robots_pkt.add()
            rep_rob.position.robot_id = i
            rep_rob.position.x = robot[0]
            rep_rob.position.y = robot[1]
            rep_rob.position.orientation = robot[2]
            rep_rob.yellowteam = False
            rep_rob.turnon = True

        for i, robot in enumerate(placement_pos["yellow_robots_pos"]):
            rep_rob = robots_pkt.add()
            rep_rob.position.robot_id = i
            rep_rob.position.x = robot[0]
            rep_rob.position.y = robot[1]
            rep_rob.position.orientation = robot[2]
            rep_rob.yellowteam = True
            rep_rob.turnon = True

        # send commands
        data = pkt.SerializeToString()
        self.com_socket.sendto(data, self.com_address)

    def get_frame(self):
        """Receive package and decode."""
        data, _ = self.vision_sock.recvfrom(1024)
        decoded_data = packet_pb2.Environment().FromString(data)
        frame = FrameVSSPB()
        frame.parse(decoded_data)
        return frame

    def send_commands(self, commands):
        # prepare commands
        pkt = packet_pb2.Packet()
        d = pkt.cmd.robot_commands

        # send wheel speed commands for each robot
        for cmd in commands:
            robot = d.add()
            robot.id = cmd.id
            robot.yellowteam = cmd.yellow

            # convert from linear speed to angular speed
            robot.wheel_left = cmd.v_wheel0
            robot.wheel_right = cmd.v_wheel1

        # send commands
        data = pkt.SerializeToString()
        self.com_socket.sendto(data, self.com_address)

    def _placement_dict_from_frame(self, frame: FrameVSSPB):
        replacement_pos: Dict[str, np.ndarray] = {}

        ball_pos: List[float] = [
            frame.ball.x,
            frame.ball.y,
            frame.ball.v_x,
            frame.ball.v_y,
        ]
        replacement_pos["ball_pos"] = np.array(ball_pos)

        blue_pos: List[List[float]] = []
        for robot in frame.robots_blue.values():
            robot_pos: List[float] = [robot.x, robot.y, robot.theta]
            blue_pos.append(robot_pos)
        replacement_pos["blue_robots_pos"] = np.array(blue_pos)

        yellow_pos: List[List[float]] = []
        for robot in frame.robots_yellow.values():
            robot_pos: List[float] = [robot.x, robot.y, robot.theta]
            yellow_pos.append(robot_pos)
        replacement_pos["yellow_robots_pos"] = np.array(yellow_pos)

        return replacement_pos
