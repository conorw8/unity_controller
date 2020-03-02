#!/usr/bin/env python

import rospy
import math
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import Pose
from bicycle import Bicycle
from unity_controller.msg import AgentVelocity, AgentPose, Velocity
from pid import PID
import subprocess, re
from kafka import KafkaProducer
from json import dumps
from time import sleep
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# TODO:
# a.) convert this to ROS and Unity
# b.) get network data node working
# c.) record new data set
# d.) connect simulation to cloud

ideal_pose = None
faulty_pose = None
residual = np.empty(3)

def loadScaler():
    path = '/home/ace/catkin_ws/src/unity_controller/data/sim_data.csv'
    df = pd.read_csv(path)
    df = df[['%velocity','%steering','%x','%y','%theta','%iteration','%time','%delay','%label']]
    df_array = df.to_numpy()
    dataset = df_array[:, :-1]
    labels = np.reshape(df_array[:, -1], (-1, 1))
    print(labels)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset = scaler.fit_transform(dataset)

    return scaler

def poseCallback(msg):
    global ideal_pose, faulty_pose, residual
    # print(msg)
    ideal_pose = msg.agents[0]
    faulty_pose = msg.agents[1]
    residual[0] = faulty_pose.position.x - ideal_pose.position.x
    residual[1] = faulty_pose.position.y - ideal_pose.position.y
    residual[2] = faulty_pose.orientation.z - ideal_pose.orientation.z

def processData(pid, line, scaler, acquire_data):
    global ideal_pose, faulty_pose
    # initialize node
    rospy.init_node('fault_data', anonymous = True)

    labels = ["ideal", "no fault", "left fault", "right fault"]
    rospy.Subscriber('/lilbot_EFD047/agent_poses', AgentPose, poseCallback, queue_size=1, tcp_nodelay=True)
    velocity_publisher = rospy.Publisher('agent_velocities', AgentVelocity, queue_size=1, tcp_nodelay=True)
    rate = rospy.Rate(10) # 10hz
    msg = AgentVelocity()
    ideal_velocity = Velocity()
    faulty_velocity = Velocity()
    msg.agent_velocities = [ideal_velocity, faulty_velocity]

    previous_time = rospy.get_time()
    time = 0.0
    start_time = 0.0
    iteration = 0
    num_features = 8
    feature_vector = np.empty(num_features)
    training_data = []
    hostname = "192.168.1.151"

    producer = KafkaProducer(bootstrap_servers=['192.168.1.108:9092'],
                             value_serializer=lambda x:
                             dumps(x).encode('utf-8'))

    while not rospy.is_shutdown():
        if ideal_pose is not None:
            # Compute PID
            time = rospy.get_time() - start_time
            dt = rospy.get_time() - previous_time

            ideal_distance, ideal_heading = pid.computeNormalDistance(ideal_pose.position.x, ideal_pose.position.y, ideal_pose.orientation.z, line)
            ideal_velocity.velocity, ideal_velocity.steering = pid.calculatePID(ideal_distance, ideal_heading, dt)
            faulty_distance, faulty_heading = pid.computeNormalDistance(faulty_pose.position.x, faulty_pose.position.y, faulty_pose.orientation.z, line)
            faulty_velocity.velocity, faulty_velocity.steering = pid.calculatePID(faulty_distance, faulty_heading, dt)

            # Check network status
            output = subprocess.Popen(["sudo", "ping",hostname, "-c", "1", "-i", "0.1"],stdout = subprocess.PIPE).communicate()[0]
            delay = re.findall(r"[0-9]+\.[0-9]+/([0-9]+\.[0-9]+)/[0-9]+\.[0-9]+/[0-9]+\.[0-9]+", output.decode('utf-8'))

            if delay == None:
                delay = np.array([0])

            feature_vector = np.array([float(faulty_pose.position.z), float(faulty_velocity.velocity + faulty_velocity.steering), float(residual[0]), float(residual[1]), float(residual[2]), float(iteration), float(time), float(delay[0])])
            feature_vector = np.reshape(feature_vector, (1, num_features))

            if acquire_data:
                sample = np.concatenate((feature_vector, np.reshape([3.0], (1, 1))), axis=1)
                training_data.append(sample)
                print(feature_vector)
            else:
                data = np.reshape(feature_vector, (1, num_features))
                normalized_data = scaler.transform(data)

                normalized_data = np.concatenate((normalized_data, np.reshape([3.0], (1, 1))), axis=1)
                print(normalized_data.tolist)

                value = {'signal' : normalized_data.tolist()}
                print(value)
                producer.send('data', value=value)
                sleep(0.05)

            previous_time = rospy.get_time()
            iteration += 1
        else:
            start_time = rospy.get_time()

        if iteration == 225:
            if len(training_data) != 0:
                training_data = np.array(training_data)
                training_data = np.reshape(training_data, (225, 9))
                if np.isnan(np.sum(training_data)):
                    return
                else:
                    f = open('/home/ace/catkin_ws/src/unity_controller/data/sim_data.csv', 'a')
                    np.savetxt(f, training_data, delimiter=",")
                    return
            else:
                return

        velocity_publisher.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    # Create a MinMaxScaler
    # scaler = loadScaler()
    scaler = None

    # Process Data
    init_pose = np.array([0.0, 0.0, math.radians(0)])
    init_pose = np.reshape(init_pose, (3,1))
    k = [0.5, 0.0, 0.0, 1, 0.0, 0.0]
    pid = PID(k)
    line = np.array([1.0, -2.0, 4.0])
    processData(pid, line, scaler, 1)

    # data = np.concatenate((path1_error, path2_error, path3_error), axis=0)
    # # print(data)
    # # f = open('/home/conor/catkin_ws/src/unity_controller/data/sim_data.csv', 'a')
    # # np.savetxt(f, data, delimiter=",")
