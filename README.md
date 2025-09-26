Time to get jiggy 

- tsting of opencv
- implementation of rgb camera and given opencv values 

find the bounds of your map
ros2 topic echo /map_metadata --once
note: resolution (m/cell), width, height, and origin.position.{x,y}

ros2 topic echo /global_costmap/costmap --once
look at msg.info.resolution, msg.info.width, msg.info.height, msg.info.origin.position.{x,y}


1. Downloading of open cv

sudo apt update
sudo apt install python3-opencv
sudo apt install python3-pip
pip3 install opencv-python
pip3 install opencv-contrib-python

2. Launch the sim
source ~/41068_ws/install/setup.bash

ros2 launch 41068_ignition_bringup 41068_ignition.launch.py slam:=true nav2:=true rviz:=true world:=seeder

3. add this for 2d goal estimate under tool properites topic 
/slam_toolbox/initialpose




4. Run the node
cd 41068_ws
ros2 run project_seeder tree_goals
ros2 run project_seeder colour_detection


Building lines

cd /home/student/41068_ws
colcon build --packages-select 41068_ignition_bringup
source install/setup.bash