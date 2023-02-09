# Import libraries
import csv
import numpy as np

class Read:
    def __init__(self, path):
        self.path = path
        self.frames = []
        self.odometry = []
        self.position = []
        self.has_goal = []
        self.goal_bounding_box = []
        self.timestamps = []
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    # print("Column names are ", ", ".join(row))
                    line_count += 1
                else:
                    self.frames.append(row[0])
                    self.robotId = row[1]
                    self.odometry.append([float(row[2]), float(row[3]), float(row[4])])
                    self.position.append([float(row[5]), float(row[6]), float(row[7])])
                    self.has_goal.append(bool(row[8]))
                    self.goal_bounding_box.append([float(row[9]), float(row[10]), float(row[11]), float(row[12])])
                    self.timestamps.append(float(row[13]))
                    line_count += 1

    def get_odometry(self):
        return np.array(self.odometry)
    
    def get_odometry_2d(self):
        return np.array(self.odometry)[:,0:2]

    def get_odometry_vectors(self):
        odm = self.get_odometry()
        return np.array([odm[:, 0] + np.cos(odm[:, 2]), odm[:, 1] + np.sin(odm[:, 2])]).T

    def get_position(self):
        return np.array(self.position)

    def get_position_2d(self):
        return np.array(self.position)[:,0:2]

    def get_position_vectors(self):
        vis = self.get_position()
        return np.array([vis[:, 0] + np.cos(vis[:, 2]), vis[:, 1] + np.sin(vis[:, 2])]).T

    def get_first_frame(self):
        return self.frames[0]

    def get_frames(self):
        return np.array(self.frames)
    
    def get_timestamps(self):
        return np.array(self.timestamps)
    
    def get_goals(self):
        return np.array(self.goal_bounding_box)

    def get_path(self):
        return self.path

    def get_steps(self):
        '''
        Result: pckt_count[n] == sum(steps[:n+1]) + pckt_count[0]
        '''
        steps = []
        for i in range (0, len(self.get_frames())):
            step = i
            steps.append(step)
        return np.array(steps)

    def get_odometry_movement(self, degrees=False, local=False):
        '''
        Result: odometry[n] == sum(odometry_movement[:n+1]) + odometry[0]
        '''
        odometry_movement = [[0,0,0]]
        odometry = self.get_odometry()
        for i in range(1,len(odometry)):
            movement = list(odometry[i] - odometry[i-1])
            if degrees: movement[2] = np.degrees(movement[2])
            odometry_movement.append(movement)
        return np.array(odometry_movement)

    def get_position_movement(self, degrees=False):
        '''
        Result: position[n] == sum(position_movement[:n+1]) + position[0]
        '''
        position_movement = [[0,0,0]]
        position = self.get_position()
        for i in range(1,len(position)):
            movement = list(position[i] - position[i-1])
            if degrees: movement[2] = np.degrees(movement[2])
            position_movement.append(movement)
        return np.array(position_movement)
    
    def rotate_to_local(self, global_x, global_y, robot_w):
        local_x = global_x*np.cos(robot_w) + global_y*np.sin(robot_w)
        local_y = -global_x*np.sin(robot_w) + global_y*np.cos(robot_w)
        return local_x, local_y

    def get_timesteps(self):
        '''
        Result: timestamps[n] == timesteps[:n+1] + timestamps[0]
        '''
        timesteps = [0]
        timestamps = self.get_timestamps()
        for i in range(1,len(timestamps)):
            # import pdb;pdb.set_trace()
            timestep = timestamps[i] - timestamps[i-1]
            timesteps.append(timestep)
        return np.array(timesteps)
    
    def get_timesteps_average(self):
        timesteps = self.get_timesteps()
        return np.mean(timesteps)
        
            
if __name__ == "__main__":
    import os

    cwd = os.getcwd()

    quadrado_nr = 1
    path = cwd+f'/localization_data/quadrado{quadrado_nr}/log.csv'
    file = Read(path)

    print(file.get_timesteps_average())

