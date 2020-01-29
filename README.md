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
8. cd ~/catkin_ws
9. catkin_make

## Unity Side

1. Install ROS# version 1.6 https://github.com/siemens/ros-sharp/releases
2. Drag and drop RosSharp folder into Assets folder in Unity
3. Make sure .NET 4.x Equivalent is enabled under edit/project settings/Player/Other Settings/Configuration


