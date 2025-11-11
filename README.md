Welcome to project seeder 

1. Downloading 

sudo apt update
sudo apt install python3-opencv
sudo apt install python3-pip
pip3 install opencv-python
pip3 install opencv-contrib-python

sudo apt install ros-humble-slam-toolbox
sudo apt install ros-humble-xacro

2. Launch the sim
source ~/41068_ws/install/setup.bash

ros2 launch 41068_ignition_bringup 41068_ignition.launch.py slam:=true nav2:=true rviz:=true world:=seeder

3. Ran through GUI (can be run as separate nodes)
- Autonomous mode
- Manual mode 

4. Run the individual node
cd 41068_ws
ros2 run project_seeder tree_goals
ros2 run project_seeder colour_detection
ros2 run project_seeder gui



5. Building lines

cd /home/student/41068_ws
colcon build --packages-select 41068_ignition_bringup
source install/setup.bash

colcon build --packages-select project_seeder
. install/setup.bash

