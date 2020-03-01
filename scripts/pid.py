import math
import numpy as np

class PID():
    def __init__(self, k):
        self.k = k
        self.velocity = 0
        self.steering = 0
        self.last_d = 0
        self.last_theta = 0
        self.Integral_d = 0
        self.Integral_theta = 0
        self.d_max = 1
        self.theta_max = math.pi
        self.max_vel = 10.0
        self.max_rad = 45.0

    def calculatePID(self, distance, heading, dt):
        self.Integral_d = self.Integral_d + distance
        self.Integral_d = np.clip(self.Integral_d, -1*self.d_max, self.d_max)
        self.Integral_theta = self.Integral_theta + heading
        self.Integral_theta = np.clip(self.Integral_theta, -1*self.theta_max, self.theta_max)

        self.velocity = self.k[0]*distance + self.k[1]*self.Integral_d + self.k[2]*(distance - self.last_d)*dt
        self.steering = self.k[3]*heading + self.k[4]*self.Integral_theta + self.k[5]*(heading - self.last_theta)*dt

        self.last_d = distance
        self.last_theta = heading

        return self.velocity, self.steering

    def angdiff(self, a, b):
        diff = a - b

        if diff < 0.0:
            diff = (diff % (-2*math.pi))
            # print(diff)
            if diff < (-math.pi):
                diff = diff + 2*math.pi
        else:
            diff = (diff % 2*math.pi)
            if diff > math.pi:
                diff = diff - 2*math.pi

        return diff

    def computeNormalDistance(self, x, y, theta, line):
        slope_of_line = math.atan2(line[0], -line[1])
        heading_error = self.angdiff(slope_of_line, theta)
        distance_error = ((x*line[0])+(y*line[1])+(line[2]))/math.sqrt((line[0]**2)+(line[1]**2))

        # print("Slope of line: %s" % slope_of_line)
        # print("Yaw: %s" % self.x[2, 0])
        # print("Line Heading: %s" % heading_error)
        # print("Distance Error: %s" % distance_error)

        return distance_error, heading_error
