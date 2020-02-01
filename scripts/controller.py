#!/usr/bin/env python

import rospy
import math
import numpy as np
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

    def odomCallback(self, msg):
        self.x = float(msg.pose.position.x)
        self.y = float(msg.pose.position.y)
        self.z = float(msg.pose.position.z)
        orientation_quaternion = msg.pose.orientation
        orientation_list = [orientation_quaternion.x, orientation_quaternion.y, orientation_quaternion.z, orientation_quaternion.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.roll = float(roll)
        self.pitch = float(pitch)
        self.yaw = float(yaw)

    ####
    #    Parameters: target = [x*, y*]
    #    Returns: Euclidean Distance Error, Heading Error
    ####
    def calculateError(self, target):
        delta_x = np.clip(target[0] - self.x, -1e50, 1e50)
        delta_y = np.clip(target[1] - self.y, -1e50, 1e50)
        desired_heading = math.atan2(delta_y, delta_x)

        self.heading = desired_heading - self.yaw
        delta_x2 = delta_x**2
        delta_y2 = delta_y**2
        if math.isinf(delta_x2):
            delta_x2 = 1e25
        if math.isinf(delta_y2):
            delta_y2 = 1e25
        self.distance = math.sqrt(delta_x2 + delta_y2)

    def moveToTarget(self, target, usePID=True, verbose=True):
        # initialize node
        rospy.init_node('mouseToJoy', anonymous = True)

        #### Setup Odometry Subscriber
        rospy.Subscriber(self.odom, PoseStamped, self.odomCallback, queue_size=5, tcp_nodelay=True)

    	#### Setup Velocity Publisher
        velocityPublisher = rospy.Publisher(self.cmd_vel, Twist, queue_size=5, tcp_nodelay=True)
        rate = rospy.Rate(20) # 20hz
        msg = Twist()
        self.time = rospy.Time.now().to_sec()
        self.iteration = 0

        while not rospy.is_shutdown():
            self.calculateError(target)
            current_time = rospy.Time.now().to_sec()
            dt = current_time - self.time
            self.pid.calculatePID(self.distance, self.heading, dt)
            self.time = current_time

            if usePID:
                self.velocity = self.pid.velocity
                self.steering = self.pid.steering

            self.path_data.append([self.x, self.y, self.z, self.roll, self.pitch, self.yaw, self.time, self.iteration])

            msg.linear.x = self.velocity
            msg.angular.z = self.steering

            if verbose:
                print("Current state: [%s, %s, %s]" % (self.x, self.y, self.yaw))

            if (self.distance <= self.eta):
                msg.linear.x = 0.0
                msg.angular.z = 0.0
                print("Target Reached")
                velocityPublisher.publish(msg)
                rate.sleep()
                return

            self.iteration += 1
            velocityPublisher.publish(msg)
            rate.sleep()
