#!/usr/bin/env python

import rospy
import math
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import Pose
from bicycle import Bicycle
from unity_controller.msg import AgentVelocity, AgentPose

# TODO:
# a.) convert this to ROS and Unity
# b.) get network data node working
# c.) record new data set
# d.) connect simulation to cloud
ideal_velocity = None
faulty_velocity = None

def inputsCallback(msg):
    global ideal_velocity, faulty_velocity
    # print(msg)
    ideal_velocity = msg.agent_velocities[0]
    faulty_velocity = msg.agent_velocities[1]

def driveClosedLoop(ideal_ugv, faulty_ugv):
    global ideal_velocity, faulty_velocity
    # initialize node
    rospy.init_node('fault_ugvs', anonymous = True)

    i = 0
    start_point = ideal_ugv.x.copy()
    labels = ["ideal", "no fault", "left fault", "right fault"]
    rospy.Subscriber('agent_velocities', AgentVelocity, inputsCallback, queue_size=1, tcp_nodelay=True)
    agents_publisher = rospy.Publisher('agent_poses', AgentPose, queue_size=1, tcp_nodelay=True)
    rate = rospy.Rate(10) # 10hz
    msg = AgentPose()
    ideal_pose = Pose()
    faulty_pose = Pose()
    msg.agents = [ideal_pose, faulty_pose]
    time = rospy.Time.now().to_sec()
    iteration = 0
    start_time = 0.0
    start_index = 0
    first_measurement = ideal_ugv.x
    first_move = True
    current_index = 0
    previous_time = 0.0
    velocity = np.random.uniform(0.2, 0.7, 1)

    while not rospy.is_shutdown():
        # print(ideal_ugv.x[0, 0], ideal_ugv.x[1, 0], ideal_ugv.x[2, 0])
        if ideal_velocity is not None:
            ideal_ugv.path_data.append([ideal_ugv.x[0, 0], ideal_ugv.x[1, 0], ideal_ugv.x[2, 0]])
            faulty_ugv.path_data.append([faulty_ugv.x[0, 0], faulty_ugv.x[1, 0], faulty_ugv.x[2, 0]])

            # Compute Bicycle model equations
            iteration = current_index - start_index
            dt = rospy.get_time() - previous_time
            # ideal ugv dynamics
            ideal_steering = ideal_velocity.velocity + ideal_velocity.steering
            ideal_steering = np.clip(ideal_steering, (-1*math.radians(ideal_ugv.max_rad)), math.radians(ideal_ugv.max_rad))
            # print(dt)
            ideal_ugv.dynamics(velocity, ideal_steering, dt)
            ideal_pose.position.x = ideal_ugv.x[0, 0]
            ideal_pose.position.y = ideal_ugv.x[1, 0]
            ideal_pose.orientation.z = ideal_ugv.x[2, 0]
            #faulty ugv dynamics
            faulty_steering = faulty_velocity.velocity + faulty_velocity.steering
            faulty_steering = np.clip(faulty_steering, (-1*math.radians(faulty_ugv.max_rad)), math.radians(faulty_ugv.max_rad))
            # print(faulty_steering)
            faulty_ugv.dynamics(velocity, faulty_steering, dt)
            faulty_pose.position.x = faulty_ugv.x[0, 0]
            faulty_pose.position.y = faulty_ugv.x[1, 0]
            faulty_pose.position.z = velocity
            faulty_pose.orientation.x = faulty_ugv.fault
            faulty_pose.orientation.z = faulty_ugv.x[2, 0]
            iteration += 1
            i += 1

        if i == ideal_ugv.max_iter:
            ideal_path = np.asarray(ideal_ugv.path_data)
            faulty_path = np.asarray(faulty_ugv.path_data)
            t = np.arange(ideal_path.shape[0])
            t = t*0.05
            line = np.array([1.0, -2.0, 4.0])
            y = (t*(-line[0]) - line[2])/line[1]
            y = np.array(y)
            plt.plot(t[:], y[:], color='black')
            plt.plot(ideal_path[:, 0], ideal_path[:, 1], color='red', label=labels[ideal_ugv.fault])
            plt.plot(faulty_path[:, 0], faulty_path[:, 1], color='green', label=labels[ideal_ugv.fault])
            plt.xlabel("x (meters)")
            plt.ylabel("y (meters)")
            plt.title("UGV Path: Problem 1.b)")
            plt.scatter(start_point[0], start_point[1], marker='o', color='blue')
            plt.show()

            return

        previous_time = rospy.get_time()
        agents_publisher.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    # Problem 1.a)
    init_pose = np.array([0.0, 0.0, math.radians(0)])
    init_pose = np.reshape(init_pose, (3,1))
    healthy_k = [0.5, 0.0, 0.0, 1, 0.0, 0.0]
    faulty_k = [0.5, 0.0, 0.0, 1, 0.0, 0.0]
    healthy_ugv = Bicycle(init_pose, 22.5, 10, healthy_k, fault=0)
    healthy_ugv.setNoiseFunction()
    init_pose = np.array([0.0, 0.0, math.radians(0)])
    init_pose = np.reshape(init_pose, (3,1))
    fault1_ugv = Bicycle(init_pose, 22.5, 10, faulty_k, fault=1)
    fault1_ugv.setNoiseFunction()

    line = np.array([1.0, -2.0, 4.0])
    healthy_ugv.x = np.array([[0.0], [0.0], [math.radians(0)]])
    healthy_ugv.path_data = []
    driveClosedLoop(healthy_ugv, fault1_ugv)
