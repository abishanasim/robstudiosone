Welcome to project seeder 

1. Downloading steps

sudo apt update
sudo apt install python3-opencv
sudo apt install python3-pip
pip3 install opencv-python
pip3 install opencv-contrib-python

sudo apt install ros-humble-slam-toolbox
sudo apt install ros-humble-xacro

sudo apt install -y python3-tk python3-pil python3-pil.imagetk

2. Enter Main branch
- Code is placed within the project_seeder file

3. Launch the sim
source /opt/ros/humble/setup.bash
cd ~/41068_ws
colcon build --symlink-install
source ~/41068_ws/install/setup.bash
 
ros2 launch 41068_ignition_bringup 41068_ignition.launch.py slam:=true nav2:=true rviz:=true world:=seeder

4. Ran through GUI (can be run as separate nodes)
- Autonomous mode
- Manual mode 

5. Run the individual node

colcon build --symlink-install
source ~/41068_ws/install/setup.bash

# Tree goal navigation node
ros2 run project_seeder tree_goals

# Color detection node
ros2 run project_seeder colour_detection

# GUI for control and visualization
ros2 run project_seeder gui

# Manual Control node
Ros2 run project_seeder manual_nav


---- Additional Steps ----

6. Building lines

cd /home/student/41068_ws
colcon build --packages-select 41068_ignition_bringup
source install/setup.bash

colcon build --packages-select project_seeder
. install/setup.bash

