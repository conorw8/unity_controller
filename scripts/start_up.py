#!/usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt
from controller import Controller

if __name__ == '__main__':
    init_position = np.array([0.0, 0.0, 0.0])
    init_orientation = np.array([0.0, 0.0, 0.0])
    odomTopic = "/robot_odom"
    velTopic = "/cmd_vel"
    # PID Gain Parameters: index 0-2 = velocity, index 3-5 = steering
    k = [0.5, 0.0, 0.0, 100.0, 0.0, 0.0]
    target = np.array([10.0, 10.0])

    turtlebot = Controller(init_position, init_orientation, odomTopic, velTopic, k)
    turtlebot.moveToTarget(target, usePID=True)

    achieved_path = turtlebot.path_data

    achieved_path = np.array(achieved_path)
    plt.plot(achieved_path[:, 0], achieved_path[:, 1])
    plt.show()
