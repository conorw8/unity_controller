#!/usr/bin/env python

import rospy
import math
import numpy as np
import pandas as pd
from pid import PID
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist, PoseStamped
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class Controller():
    def __init__(self, initial_position, initial_orientation, odometry_topic, velocity_topic, k):
        self.x = initial_position[0]
        self.y = initial_position[1]
        self.z = initial_position[2]
        self.roll = initial_orientation[0]
        self.pitch = initial_orientation[1]
        self.yaw = initial_orientation[2]
        self.velocity = 0.0
        self.steering = 0.0
        self.odom = odometry_topic
        self.cmd_vel = velocity_topic
        self.distance = 1000000.0
        self.heading = 0.0
        self.pid = PID(k)
        self.time = 0.0
        self.iteration = 0
        self.eta = 0.01
        self.path_data = []
        self.noise_distribution = [[ 2.00000000e+01, 6.62201978e-03, -1.99941291e+01],
                                   [-5.10091508, -0.06503655, 5.08946158],
                                   [-0.01754703, 1.0, 0.06324038]]
        self.noise_a = 0.0
        self.noise_b = 0.0
        self.noise_c = 0.0
        self.white_noise = 0.0
        self.normalDistance = 0.0

    def odomCallback(self, msg):
        noise = self.computeNoiseFunction()
        # print("noise function: %s" % noise)
        self.x = float(msg.pose.position.x)
        self.y = float(msg.pose.position.y) + noise
        self.z = float(msg.pose.position.z)
        orientation_quaternion = msg.pose.orientation
        orientation_list = [orientation_quaternion.x, orientation_quaternion.y, orientation_quaternion.z, orientation_quaternion.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.roll = float(roll)
        self.pitch = float(pitch)
        self.yaw = float(yaw)

    def readNoiseFunction(self, path):
        df = pd.read_csv(path)
        dataset = df[['mean1','mean2','mean3','std1','std2','std3']]
        dataset = dataset.to_numpy()

        dataset = np.concatenate([np.reshape(np.zeros(dataset.shape[1]), (1, -1)), dataset])
        print(dataset.shape)

        self.noise_distribution = dataset

    def setNoiseFunction(self, fault):
        a_mu = self.noise_distribution[fault, 0]
        b_mu = self.noise_distribution[fault, 1]
        c_mu = self.noise_distribution[fault, 2]
        a_sigma = 0.01
        b_sigma = 0.02
        c_sigma = 0.04

        self.noise_a = np.random.normal(a_mu, a_sigma, 1)
        self.noise_b = np.random.normal(b_mu, b_sigma, 1)
        self.noise_c = np.random.normal(c_mu, c_sigma, 1)

        print(self.noise_a, self.noise_b, self.noise_c)

        if fault != 0:
            self.white_noise = 0.1

    def computeNoiseFunction(self):
        fault_noise = self.noise_a * np.exp(self.noise_b * self.x) + self.noise_c
        white_noise = np.random.normal(0, self.white_noise)

        return fault_noise + white_noise

    def angdiff(self, a, b):
        diff = a - b
        print("diff: %s" % diff)
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

    ####
    #    Parameters: target = [x*, y*]
    #    Returns: Euclidean Distance Error, Heading Error
    ####
    def calculateError(self, target, fault):
        delta_x = target[0] - self.x
        delta_y = target[1] - self.y
        desired_heading = math.atan2(delta_y, delta_x)
        self.heading = self.angdiff(desired_heading, self.yaw)

        delta_x2 = delta_x**2
        delta_y2 = delta_y**2

        self.distance = math.sqrt(delta_x2 + delta_y2)

    def computeNormalDistance(self, line, fault):
        slope_of_line = math.atan2(line[0], -line[1])
        self.heading = self.angdiff(slope_of_line, self.yaw)
        self.distance = ((self.x*line[0])+(self.y*line[1])+(line[2]))/math.sqrt((line[0]**2)+(line[1]**2))
