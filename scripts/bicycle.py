import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from pid import PID

class Bicycle():
    def __init__(self, initial_pose, t, rate, k, fault):
        self.x = initial_pose
        self.k = k
        self.max_rad = 45.0
        self.max_vel = 4.5
        self.L = 0.19
        self.t = t
        self.rate = rate
        self.max_iter = t / (1.0 / rate)
        self.pid = PID(self.k)
        self.path_data = []
        self.heading_errors = []
        self.x_dots = []
        self.fault = fault
        self.velocity = 0.0
        self.steering = 0.0
        self.P = np.array([[0.0, 0.0, 0.0],
                                            [5.00000000e+00, 2.58942535e-02, -4.99349846e+00],
                                            [1.29143976, 0.20432189, -1.29298698],
                                            [-0.24916725, 0.28371769, 0.2538033]])
        self.Q = np.array([[0.0, 0.0, 0.0],
                           [1.47100269e-10, 4.53318633e-03, 9.43288971e-04],
                           [0.14536192, 0.02894827, 0.14429605],
                           [0.05916725, 0.16029422, 0.05427143]])
        self.noise_a = 0.0
        self.noise_b = 0.0
        self.noise_c = 0.0
        self.white_noise = 0.0

    def setNoiseFunction(self):
        a_mu = self.P[int(self.fault), 0]
        b_mu = self.P[int(self.fault), 1]
        c_mu = self.P[int(self.fault), 2]
        a_sigma = self.Q[int(self.fault), 0]
        b_sigma = self.Q[int(self.fault), 1]
        c_sigma = self.Q[int(self.fault), 2]

        if self.fault == 0:
            self.noise_a = 0.0
            self.noise_b = 0.0
            self.noise_c = 0.0
            self.white_noise = 0.0
        else:
            self.noise_a = np.random.normal(a_mu, a_sigma, 1)
            self.noise_b = np.random.normal(b_mu, b_sigma, 1)
            self.noise_c = np.random.normal(c_mu, c_sigma, 1)
            self.white_noise = 0.15

        self.noise_b = np.clip(self.noise_b, -0.75, 0.75)
        print("[%s, %s, %s]" % (self.noise_a, self.noise_b, self.noise_c))

    def computeNoiseFunction(self):
        fault_noise = self.noise_a * np.exp(self.noise_b * self.x[0, 0]) + self.noise_c
        white_noise = np.random.normal(0, self.white_noise)

        return fault_noise + white_noise

    def dynamics(self, v, gamma, dt):
        #dynamics
        yaw_dot = ((v/self.L)*(math.tan(gamma)))*dt
        x_dot = (v * math.cos(self.x[2, 0]) - math.sin(self.x[2, 0])*self.computeNoiseFunction()*0.1)*dt
        y_dot = (v * math.sin(self.x[2, 0]) + math.cos(self.x[2, 0])*self.computeNoiseFunction()*0.1)*dt

        #derivatives
        self.x[2, 0] = self.x[2, 0] + yaw_dot
        self.x[0, 0] = self.x[0, 0] + x_dot
        self.x[1, 0] = self.x[1, 0] + y_dot

    def angdiff(self, a, b):
        diff = a - b
        if diff < 0.0:
            diff = (diff % (-2*math.pi))
            print(diff)
            if diff < (-math.pi):
                diff = diff + 2*math.pi
        else:
            diff = (diff % 2*math.pi)
            if diff > math.pi:
                diff = diff - 2*math.pi

        return diff

    def calculateError(self, target):
        delta_x = target[0] - self.x[0, 0]
        delta_y = target[1] - self.x[1, 0]
        desired_heading = math.atan2(delta_y, delta_x)
        heading_error = self.angdiff(desired_heading, self.x[2, 0])

        delta_x2 = delta_x**2
        delta_y2 = delta_y**2

        distance_error = math.sqrt(delta_x2 + delta_y2)
        # print("distance: %s" % distance_error)

        return distance_error, heading_error

    def computeNormalDistance(self, line):
        slope_of_line = math.atan2(line[0], -line[1])
        heading_error = self.angdiff(slope_of_line, self.x[2, 0])
        distance_error = ((self.x[0, 0]*line[0])+(self.x[1, 0]*line[1])+(line[2]))/math.sqrt((line[0]**2)+(line[1]**2))

        # print("Slope of line: %s" % slope_of_line)
        # print("Yaw: %s" % self.x[2, 0])
        # print("Line Heading: %s" % heading_error)
        # print("Distance Error: %s" % distance_error)

        return distance_error, heading_error
