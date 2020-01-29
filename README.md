# unity_controller

# Installation and Setup

## Linux Side

1. Install in ~/catkin_ws/src/ directory
2. cd ~/catkin_ws
3. catkin_make
4. Install ROS# version 1.6 https://github.com/siemens/ros-sharp/releases
5. Copy the following folders into ~/catkin_ws/src/:
  - file_server
  - gazebo_simulation_scene
  - unity_simulation_scene
6. `chmod +x ~/catkin_ws/src/gazebo_simulation_scene/scripts/joy_to_twist.py`
7. `chmod +x ~/catkin_ws/src/unity_simulation_scene/scripts/mouse_to_joy.py`
8. `chmod +x ~/catkin_ws/src/unity_controller/scripts/controller.py`
9. cd ~/catkin_ws
10. catkin_make
11. Add the IP address of the host running Unity to your hosts file as well as a generic name for it.
  - `sudo nano /etc/hosts`
  - `192.168.1.XXX NewPC`

## Unity Side

1. Install ROS# version 1.6 https://github.com/siemens/ros-sharp/releases
2. Drag and drop RosSharp folder into Assets folder in Unity
3. Make sure .NET 4.x Equivalent is enabled under edit > project settings > Player > Other Settings > Configuration

# Import Robot URDF Model

## Linux Side

1. `roslaunch file_server publish_description_turtlebot2.launch`

## Unity Side

1. RosBridgeClient > Transfer URDF from ROS...
2. Change the Address field to match the IP address of the Linux machine.
3. Under Settings > Timeout, increase the value to 100
4. Press Read Robot Description
5. Do you want to generate a Turtlebot2 GameObject now?: Press Yes
