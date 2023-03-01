
sleep 10
export ROS_MASTER_URI=http://localhost:1135$1
export GAZEBO_MASTER_URI=http://localhost:1137$1
python $2 $3