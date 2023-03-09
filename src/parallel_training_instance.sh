tmux send-keys "export ROS_MASTER_URI=http://localhost:1135$1" Enter
tmux send-keys "export GAZEBO_MASTER_URI=http://localhost:1137$1" Enter
tmux send-keys "roslaunch fl4sr fl4sr_real_8_no_obs.launch" Enter
tmux split-window

tmux send-keys "./parallel_training_node.sh $1 $2 $3" Enter

#tmux send-keys "export ROS_MASTER_URI=http://localhost:1135$1" Enter
#tmux send-keys "export GAZEBO_MASTER_URI=http://localhost:1137$1" Enter
#tmux send-keys "python $2" Enter
