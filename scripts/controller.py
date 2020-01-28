#!/usr/bin/env python

import rospy
import math
import numpy as np
from pid import PID
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist, PoseStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class Controller():
    def __init__(self, initial_position, initial_orientation, odometry_topic, velocity_topic, k):
        self.x = initial_position[0]
        self.y = initial_position[1]
        self.z = initial_position[2]
        self.roll = initial_orientation[0]
        self.pitch = initial_orientation[1]
        self.yaw = initial_orientation[2]
        self.odom = odometry_topic
        self.cmd_vel = velocity_topic
        self.distance = 0.0
        self.heading = 0.0
        self.pid = PID(k)
        self.last_time = 0.0
        self.threshold = 0.5

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

    def moveToTarget(self):
        # initialize node
        rospy.init_node('mouseToJoy', anonymous = True)

        #### Setup Odometry Subscriber
        rospy.Subscriber(self.odom, PoseStamped, self.odomCallback, queue_size=5, tcp_nodelay=True)

    	#### Setup Velocity Publisher
        velocityPublisher = rospy.Publisher(self.cmd_vel, Twist, queue_size=5, tcp_nodelay=True)
        rate = rospy.Rate(20) # 20hz
        msg = Twist()
        target = np.array([10.0, 10.0])

        while not rospy.is_shutdown():
            self.calculateError(target)
            dt = rospy.Time.now().to_sec() - self.last_time
            self.pid.calculatePID(self.distance, self.heading, dt)
            self.last_time = rospy.Time.now().to_sec()

            print(self.x, self.y)

            if math.fabs(self.distance) > self.threshold:
                msg.linear.x = 1
                # msg.linear.x = self.pid.velociy
                # msg.angular.z = 100
                msg.angular.z = self.pid.steering
            else:
                msg.linear.x = 0.0
                msg.angular.z = 0.0
                print("Target Reached")

            velocityPublisher.publish(msg)
            rate.sleep()

if __name__ == '__main__':
    init_position = np.array([0.0, 0.0, 0.0])
    init_orientation = np.array([0.0, 0.0, 0.0])
    odomTopic = "/odom"
    velTopic = "/cmd_vel"
    # PID Gain Parameters: index 0-2 = velocity, index 3-5 = steering
    k = [0.0, 0.0, 0.0, 50.0, 0.0, 0.0]

    turtlebot = Controller(init_position, init_orientation, odomTopic, velTopic, k)
    turtlebot.moveToTarget()
