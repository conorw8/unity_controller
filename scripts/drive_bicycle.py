import numpy as np
import math
import matplotlib.pyplot as plt
from bicycle import Bicycle
from pid import PID

# TODO:
# a.) convert this to ROS and Unity
# b.) get network data node working
# c.) record new data set
# d.) connect simulation to cloud

def driveOpenLoop(ugv, color):
    print(ugv.path_data)
    i = 0

    while(i != ugv.max_iter):
        ugv.path_data.append([ugv.x[0, 0], ugv.x[1, 0], ugv.x[2, 0]])
        ugv.dynamics(0.4, 0.0, (1.0/ugv.rate))
        i += 1

    path = np.asarray(ugv.path_data)
    print(path.shape)
    plt.scatter(path[:, 0], path[:, 1], color=color)
    plt.xlabel("x (meters)")
    plt.ylabel("y (meters)")
    plt.title("UGV Path: Problem 1.a)")
    plt.show()

def driveClosedLoop(ugv, line, color):
    pid = PID(ugv.k)
    i = 0
    distance = 100
    start_point = ugv.x.copy()
    labels = ["ideal", "no fault", "left fault", "right fault"]

    while((i != ugv.max_iter)):
        print(ugv.x[0, 0], ugv.x[1, 0], ugv.x[2, 0], float(i), ugv.fault)
        ugv.path_data.append([ugv.x[0, 0], ugv.x[1, 0], ugv.x[2, 0], float(i), ugv.fault])
        distance, heading = ugv.computeNormalDistance(line)
        pid.calculatePID(distance, heading, (1.0/ugv.rate))
        steering = pid.steering + pid.velocity
        # print(pid.steering, pid.velocity)
        # print(steering)
        steering = np.clip(steering, (-1*math.radians(45)), math.radians(45))
        # print("Steering: %s" % steering)
        ugv.dynamics(0.4, steering, (1.0/ugv.rate))
        i += 1

    path = np.asarray(ugv.path_data)

    plt.plot(path[:, 0], path[:, 1], color=color, label=labels[ugv.fault])
    plt.xlabel("x (meters)")
    plt.ylabel("y (meters)")
    plt.title("UGV Path: Problem 1.b)")
    plt.scatter(start_point[0], start_point[1], marker='o', color='blue')

if __name__ == '__main__':
    # Problem 1.a)
    init_pose = np.array([0.0, 0.0, math.radians(0)])
    init_pose = np.reshape(init_pose, (3,1))
    healthy_k = [0.5, 0.0, 0.0, 1, 0.0, 0.0]
    faulty_k = [0.5, 0.0, 0.0, 1, 0.0, 0.0]
    healthy_ugv = Bicycle(init_pose, 22.5, 20, healthy_k, fault=0)
    healthy_ugv.setNoiseFunction()
    init_pose = np.array([0.0, 0.0, math.radians(0)])
    init_pose = np.reshape(init_pose, (3,1))
    fault1_ugv = Bicycle(init_pose, 22.5, 20, faulty_k, fault=1)
    fault1_ugv.setNoiseFunction()
    init_pose = np.array([0.0, 0.0, math.radians(0)])
    init_pose = np.reshape(init_pose, (3,1))
    fault2_ugv = Bicycle(init_pose, 22.5, 20, faulty_k, fault=2)
    fault2_ugv.setNoiseFunction()
    init_pose = np.array([0.0, 0.0, math.radians(0)])
    init_pose = np.reshape(init_pose, (3,1))
    fault3_ugv = Bicycle(init_pose, 22.5, 20, faulty_k, fault=3)
    fault3_ugv.setNoiseFunction()
    driveOpenLoop(healthy_ugv, color='red')
    driveOpenLoop(fault1_ugv, color='blue')
    driveOpenLoop(fault2_ugv, color='green')
    driveOpenLoop(fault3_ugv, color='orange')

    line = np.array([1.0, -2.0, 4.0])
    healthy_ugv.x = np.array([[0.0], [0.0], [math.radians(0)]])
    healthy_ugv.path_data = []
    driveClosedLoop(healthy_ugv, line, color='red')

    fault1_ugv.x = np.array([[0.0], [0.0], [math.radians(0)]])
    fault1_ugv.path_data = []
    driveClosedLoop(fault1_ugv, line, color='blue')

    fault2_ugv.x = np.array([[0.0], [0.0], [math.radians(0)]])
    fault2_ugv.path_data = []
    driveClosedLoop(fault2_ugv, line, color='green')

    fault3_ugv.x = np.array([[0.0], [0.0], [math.radians(0)]])
    fault3_ugv.path_data = []
    driveClosedLoop(fault3_ugv, line, color='orange')

    xlim = 7
    t = np.arange(0, 400)
    t = t*0.05
    y = (t*(-line[0]) - line[2])/line[1]
    y = np.array(y)
    plt.plot(t[:], y[:], color='black')
    plt.ylim(0, xlim)
    plt.xlim(0, xlim)
    plt.xlabel("x (meters)")
    plt.ylabel("y (meters)")
    plt.title("UGV Path Error")
    plt.legend()
    plt.show()

    health_path = np.array(healthy_ugv.path_data)
    fault1_path = np.array(fault1_ugv.path_data)
    fault2_path = np.array(fault2_ugv.path_data)
    fault3_path = np.array(fault3_ugv.path_data)
    path1_error = fault1_path
    path2_error = fault2_path
    path3_error = fault3_path
    path1_error[:, 0:3] = fault1_path[:, 0:3] - health_path[:, 0:3]
    path2_error[:, 0:3] = fault2_path[:, 0:3] - health_path[:, 0:3]
    path3_error[:, 0:3] = fault3_path[:, 0:3] - health_path[:, 0:3]
    labels = ["ideal", "no fault", "left fault", "right fault"]
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(path1_error[:, 0], color='blue', label=labels[1])
    ax1.set_title("UGV X Error")
    ax1.set_xlabel("t")
    ax1.set_ylabel("X Residual")
    ax1.plot(path2_error[:, 0], color='green', label=labels[2])
    ax1.plot(path3_error[:, 0], color='orange', label=labels[3])
    ax1.legend()

    ax2.plot(path1_error[:, 1], color='blue', label=labels[1])
    ax2.plot(path2_error[:, 1], color='green', label=labels[2])
    ax2.plot(path3_error[:, 1], color='orange', label=labels[3])
    ax2.set_title("UGV Y Error")
    ax2.set_xlabel("t")
    ax2.set_ylabel("Y Residual")
    ax2.legend()

    ax3.plot(path1_error[:, 2], color='blue', label=labels[1])
    ax3.plot(path2_error[:, 2], color='green', label=labels[2])
    ax3.plot(path3_error[:, 2], color='orange', label=labels[3])
    ax3.set_title("UGV Theta Error")
    ax3.set_xlabel("t")
    ax3.set_ylabel("Theta Residual")
    ax3.legend()
    plt.show()

    data = np.concatenate((path1_error, path2_error, path3_error), axis=0)
    print(data)
    # f = open('/home/conor/catkin_ws/src/unity_controller/data/sim_data.csv', 'a')
    # np.savetxt(f, data, delimiter=",")
