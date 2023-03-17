#/usr/bin/bash
# Wrttien by Seongin Na
# 1. Move to the parent dir
# 2. Download dependency packages (turtlebot3, gazebo_ros_pkgs for StepControl)
# 3. export TURTLEBOT3_MODEL env variable into .bashrc
# Please build downloaded packages before running experiments. 
cd ..
git clone --branch issue_1268 git@github.com:alikureishy/gazebo_ros_pkgs.git
git clone git@github.com:ROBOTIS-GIT/turtlebot3.git
echo export TURTLEBOT3_MODEL="waffle_pi" >> ~/.bashrc

