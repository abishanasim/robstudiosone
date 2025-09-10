Time to get jiggy 

- tsting of opencv
- implementation of rgb camera and given opencv values 

1. Downloading of open cv

sudo apt update
sudo apt install python3-opencv
sudo apt install python3-pip
pip3 install opencv-python
pip3 install opencv-contrib-python

2. Launch the sim
source ~/41068_ws/install/setup.bash

ros2 launch 41068_ignition_bringup 41068_ignition.launch.py
ros2 launch 41068_ignition_bringup 41068_ignition.launch.py slam:=true nav2:=true rviz:=true

#large demo world
ros2 launch 41068_ignition_bringup 41068_ignition.launch.py world:=large_demo
ros2 launch 41068_ignition_bringup 41068_ignition.launch.py world:=large_demo

3. Run the node
cd ~robstudioone/
python3 colour_detection.py