# Import libraries
import csv
import numpy as np

class Read:
    def __init__(self, path, compare = False):
        self.path = path
        self.odometry = []
        self.vision = []
        self.motors = []
        self.pckt_count = []
        self.compare = []
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    # print("Column names are ", ", ".join(row))
                    line_count += 1
                else:
                    self.robotId = row[0]
                    self.odometry.append([float(row[1]), float(row[2]), float(row[3])])
                    self.vision.append([float(row[8]), float(row[9]), float(row[10])])
                    self.pckt_count.append(int(row[14]))
                    if compare:
                        # GYRO_W, ODM_W, BOTH_W, VIS_W
                        self.compare.append([float(row[4]), float(row[5]), float(row[6]), float(row[7])])
                    else:
                        self.motors.append([float(row[4]), float(row[5]), float(row[6]), float(row[7])])
                    line_count += 1
            # print('Processed {0} lines.'.format(line_count))

    def get_odometry(self):
        return np.array(self.odometry)
    
    def get_odometry_2d(self):
        return np.array(self.odometry)[:,0:2]

    def get_odometry_vectors(self):
        odm = self.get_odometry()
        return np.array([odm[:, 0] + np.cos(odm[:, 2]), odm[:, 1] + np.sin(odm[:, 2])]).T

    def get_vision(self):
        return np.array(self.vision)

    def get_vision_2d(self):
        return np.array(self.vision)[:,0:2]

    def get_vision_vectors(self):
        vis = self.get_vision()
        return np.array([vis[:, 0] + np.cos(vis[:, 2]), vis[:, 1] + np.sin(vis[:, 2])]).T

    def get_compare(self):
        if self.compare.Length > 0:
            return np.array(self.compare)

    def get_packet_count(self):
        return np.array(self.pckt_count)
    
    def get_motors(self):
        return np.array(self.motors)

    def get_path(self):
        return self.path

    def get_steps(self):
        '''
        Result: pckt_count[n] == sum(steps[:n+1]) + pckt_count[0]
        '''
        steps = [0]
        pckt_count = self.get_packet_count()
        for i in range (1, len(pckt_count)):
            step = (pckt_count[i]-pckt_count[i-1])
            if step<0: step=step+255    # pckts only count from 0 up to 254, so: if difference is negative -> sum 255
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

    def get_vision_movement(self, degrees=False):
        '''
        Result: vision[n] == sum(vision_movement[:n+1]) + vision[0]
        '''
        vision_movement = [[0,0,0]]
        vision = self.get_vision()
        for i in range(1,len(vision)):
            movement = list(vision[i] - vision[i-1])
            if degrees: movement[2] = np.degrees(movement[2])
            vision_movement.append(movement)
        return np.array(vision_movement)
    
    def get_vision_speeds_list(self, degrees=False):
        vision_movements = self.get_vision_movement(degrees)
        steps = self.get_steps()
        speeds = []
        time_step = 0.005   # 5 ms between packets
        for i in range(0,len(steps)):
            if steps[i]==0: vx, vy, vw = 0, 0, 0
            else: vx, vy, vw = vision_movements[i]/(steps[i]*time_step)
            speed = vx, vy, vw
            speeds.append(speed)
        return np.array(speeds)

    def get_odometry_speeds_list(self, degrees=False):
        odometry_movements = self.get_odometry_movement(degrees)
        steps = self.get_steps()
        speeds = []
        time_step = 0.005   # 5 ms between packets
        for i in range(0,len(steps)):
            if steps[i]==0: vx, vy, vw = 0, 0, 0
            else: vx, vy, vw = odometry_movements[i]/(steps[i]*time_step)
            speed = vx, vy, vw
            speeds.append(speed)
        return np.array(speeds)
    
    def rotate_to_local(self, global_x, global_y, robot_w):
        local_x = global_x*np.cos(robot_w) + global_y*np.sin(robot_w)
        local_y = -global_x*np.sin(robot_w) + global_y*np.cos(robot_w)

        return local_x, local_y
        
            
if __name__ == "__main__":
    import os

    cwd = os.getcwd()

    quadrado_nr = 10
    path = cwd+f'/odometry_data/quadrado_{quadrado_nr}.csv'
    file = Read(path)

    print(file.get_vision_speeds_list())

