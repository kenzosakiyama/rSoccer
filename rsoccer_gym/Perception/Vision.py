import cv2
import math
import numpy as np

class Camera:
    '''
    Defines camera properties
    '''
    def __init__(
                self,
                camera_matrix = np.identity(3),
                camera_distortion=np.zeros((4,1)),
                camera_to_robot_axis_offset = 100,
                camera_height = 175,
                camera_FOV = 78,
                ):

        self.intrinsic_parameters = camera_matrix
        self.distortion_parameters = camera_distortion
        self.rotation_vector: np.array((3,1)).T
        self.rotation_matrix: np.array((3,3))
        self.translation_vector: np.array((3,1)).T
        

        self.height = camera_height                 # IN MILLIMETERS
        self.offset = camera_to_robot_axis_offset   # IN MILLIMETERS
        self.FOV = camera_FOV                       # IN DEGREES

    def compute_pose_from_points(self, points3d, points2d):
        """
        Compute camera pose to object from 2D-3D points correspondences

        Solves PnP problem using OpenCV solvePnP() method assigning
        camera pose from the corresponding 2D-3D matched points.

        Parameters
        ------------
        points3d: 3D coordinates of points

        points2d: pixel positions on image
        """
        
        _,rvec,tvec=cv2.solvePnP(
                                points3d,
                                points2d,
                                self.intrinsic_parameters,
                                self.distortion_parameters
                                )                                

        rmtx, jacobian=cv2.Rodrigues(rvec)
        
        pose = cv2.hconcat((rmtx,tvec))

        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose)

        camera_position = -np.linalg.inv(rmtx)@tvec
        self.height = camera_position[2,0]

        self.rotation_vector = rvec
        self.rotation_matrix = rmtx
        self.translation_vector = tvec

class SSLEmbeddedVision:
    '''
    Class for simulating Vision Blackout vision module
    '''

    path_to_intrinsic_parameters = "/home/rc-blackout/rSoccer/rsoccer_gym/Perception/camera_matrix_C922.txt"
    path_to_points3d = "/home/rc-blackout/rSoccer/rsoccer_gym/Perception/calibration_points3d.txt"
    path_to_points2d = "/home/rc-blackout/rSoccer/rsoccer_gym/Perception/calibration_points2d.txt"
    camera_matrix = np.loadtxt(path_to_intrinsic_parameters)
    points3d = np.loadtxt(path_to_points3d, dtype="float64")
    points2d = np.loadtxt(path_to_points2d, dtype="float64")

    def __init__(
                self,
                vertical_lines_nr = 1,
                input_width = 640,
                input_height = 480,
                camera_matrix = camera_matrix,
                points3d = points3d,
                points2d = points2d
                ):
        self.camera = Camera(camera_matrix=camera_matrix)
        self.camera.compute_pose_from_points(points3d=points3d, points2d=points2d)
        self.vertical_scan_angles = []      # IN DEGREES
        for i in range(0,vertical_lines_nr):
            angle = (i+1)*self.camera.FOV/(vertical_lines_nr+1) - self.camera.FOV/2
            self.vertical_scan_angles.append(angle)
        
    def project_line(self, x, y, theta):
        coef = math.tan(math.radians(theta))
        intercept = y - coef*x
        return coef, intercept

    def intercept_upper_boundary(self, a, b, field):
        y = field.width/2 + field.boundary_width
        x = (y-b)/a
        return x, y

    def intercept_left_boundary(self, a, b, field):
        x = -field.length/2 - field.boundary_width
        y = a*x + b
        return x, y

    def intercept_lower_boundary(self, a, b, field):
        y = -field.width/2 - field.boundary_width
        x = (y-b)/a
        return x, y

    def intercept_right_boundary(self, a, b, field):
        x = field.length/2 + field.boundary_width
        y = a*x + b
        return x, y

    def get_distance(self, x1, y1, x2, y2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    def intercept_field_boundaries(self, x, y, line_dir, field):
        a, b = self.project_line(x, y, line_dir)
        if 0 == line_dir:
            x1, y1 = self.intercept_right_boundary(a, b, field)
            return x1, y1
        elif 90 == line_dir:
            x1, y1 = self.intercept_upper_boundary(a, b, field)
            return x1, y1
        elif 180 == line_dir or -180 == line_dir:
            x1, y1 = self.intercept_left_boundary(a, b, field)
            return x1, y1
        elif -90 == line_dir:
            x1, y1 = self.intercept_lower_boundary(a, b, field)
            return x1, y1
        elif 0 < line_dir and line_dir < 90:
            x1, y1 = self.intercept_right_boundary(a, b, field)
            x2, y2 = self.intercept_upper_boundary(a, b, field)
        elif 90 < line_dir and line_dir < 180:
            x1, y1 = self.intercept_left_boundary(a, b, field)
            x2, y2 = self.intercept_upper_boundary(a, b, field)
        elif -180 < line_dir and line_dir < -90:
            x1, y1 = self.intercept_left_boundary(a, b, field)
            x2, y2 = self.intercept_lower_boundary(a, b, field)
        elif -90 < line_dir and line_dir < 0:
            x1, y1 = self.intercept_right_boundary(a, b, field)
            x2, y2 = self.intercept_lower_boundary(a, b, field)

        if self.get_distance(x, y, x1, y1) < self.get_distance(x, y, x2, y2):
            return x1, y1
        else:
            return x2, y2

    def convert_xy_to_angles(self, x, y):
        '''
        Converts an x, y relative position to relative vertical and horizontal angles, as suggested in: 
            Fast and Robust Edge-Based Localization in the Sony Four-Legged Robot League - 2003
        '''
        theta_v = np.rad2deg(np.arctan2(self.camera.height, x))
        theta_h = np.rad2deg(np.arctan2(y, x))
        return [theta_v, theta_h]

    def detect_boundary_points(self, x, y, w, field):
        intercepts = []
        for angle in self.vertical_scan_angles:
            line_dir = angle + w
            line_dir = ((line_dir + 180) % 360) - 180
            interception_x, interception_y = self.intercept_field_boundaries(x, y, line_dir, field)
            interception_x, interception_y = self.convert_to_local(interception_x, interception_y, x, y, w)
            intercepts.append([interception_x, interception_y])
            # intercepts.append(self.convert_xy_to_angles(interception_x, interception_y))

        return intercepts
    
    def convert_to_local(self, global_x, global_y, robot_x, robot_y, theta):
        x = global_x - robot_x
        y = global_y - robot_y
        theta = math.radians(theta)
        x, y = x*np.cos(theta) + y*np.sin(theta),\
            -x*np.sin(theta) + y*np.cos(theta)

        return x, y

if __name__ == "__main__":
    from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv

    env = SSLBaseEnv(
        field_type=1,
        n_robots_blue=0,
        n_robots_yellow=0,
        time_step=0.025)
        
    env.field.boundary_width = 0.3

    vision = SSLEmbeddedVision(vertical_lines_nr=6)
    
    boundary_points = vision.detect_boundary_points(0, 0, 0, env.field)
    for point in boundary_points:
        print(point)
    # import pdb;pdb.set_trace()
