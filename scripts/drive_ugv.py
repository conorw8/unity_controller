#!/usr/bin/env python

import rospy
import math
import numpy as np
import matplotlib.pyplot as plt
from controller import Controller
from geometry_msgs.msg import Twist, PoseStamped

def followTarget(model, line, usePID=True, condition=3, verbose=True):
    # initialize node
    rospy.init_node('followTarget', anonymous = True)

    #### Setup Odometry Subscriber
    rospy.Subscriber(model.odom, PoseStamped, model.odomCallback, queue_size=5, tcp_nodelay=True)

	#### Setup Velocity Publisher
    velocityPublisher = rospy.Publisher(model.cmd_vel, Twist, queue_size=5, tcp_nodelay=True)
    rate = rospy.Rate(20) # 20hz
    msg = Twist()
    model.time = rospy.Time.now().to_sec()
    model.iteration = 0
    start_time = 0.0
    start_index = 0
    first_measurement = model.x
    first_move = True
    current_index = 0

    while not rospy.is_shutdown():
        if model.x != first_measurement and first_move:
            start_time = model.time
            start_index = model.iteration
            first_move = False

        model.time = model.time - start_time
        model.iteration = current_index - start_index
        model.computeNormalDistance(line, condition)
        current_time = rospy.Time.now().to_sec()
        dt = current_time - model.time
        print("current time %s" % dt)
        model.pid.calculatePID(model.distance, model.heading, dt)
        model.time = current_time - start_time

        if usePID:
            model.velocity = 0.05
            # model.steering = model.pid.steering - model.pid.velocity
            # print(model.steering)
            steering = model.pid.steering + model.pid.velocity
            model.steering = np.clip(steering, (-1*math.radians(45)), math.radians(45))

        if not first_move:
            model.path_data.append([model.x, model.y, model.z, model.roll, model.pitch, model.yaw, model.time, model.iteration])

        msg.linear.x = model.velocity
        msg.angular.z = model.steering

        if verbose and not first_move:
            #print("time iteration: %s, %s" % (self.time, self.iteration))
            print(model.distance)
            print("Current state: [%s, %s, %s]" % (model.x, model.y, model.yaw))

        if (len(model.path_data) == 800):
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            print("Target Reached")
            velocityPublisher.publish(msg)
            rate.sleep()
            return

        model.iteration += 1
        current_index += 1
        velocityPublisher.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    init_position = np.array([0.0, 0.0, 0.0])
    init_orientation = np.array([0.0, 0.0, 0.0])
    odomTopic = "/robot_odom"
    velTopic = "/cmd_vel"
    path = '/home/conor/catkin_ws/src/network_faults/data/distributions.csv'
    # PID Gain Parameters: index 0-2 = velocity, index 3-5 = steering
    k = [0.5, 0.0, 0.0, 1, 0.0, 0.0]
    line = np.array([1.0, -2.0, 4.0])

    rover = Controller(init_position, init_orientation, odomTopic, velTopic, k)
    rover.eta = 0.05
    rover.readNoiseFunction(path)
    rover.setNoiseFunction(fault=0)
    followTarget(rover, line, usePID=True, condition=0)

    achieved_path = rover.path_data

    xlim = 40
    achieved_path = np.array(achieved_path)
    plt.plot(achieved_path[:, 0], achieved_path[:, 1])
    plt.scatter(achieved_path[0, 0], achieved_path[0, 1], color='green')
    t = np.arange(0, 800)
    t = t*0.05
    y = (t*(-line[0]) - line[2])/line[1]
    y = np.array(y)
    plt.plot(t[:], y[:], color='red')
    plt.ylim(-5, xlim)
    plt.xlim(-5, xlim)
    plt.show()

    # f = open('/home/conor/catkin_ws/src/network_faults/data/model_data.csv', 'a')
    # np.savetxt(f, achieved_path, delimiter=",")
