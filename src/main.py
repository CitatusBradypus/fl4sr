#! /usr/bin/env python3.8
import sys
import os
import time
import rospy
import random
import roslaunch
import numpy as np
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')
from IndividualDDPG import IndividualDDPG
from SharedNetworkDDPG import SharedNetworkDDPG
from SharedExperienceDDPG import SharedExperienceDDPG
from FederatedLearningDDPG import FederatedLearningDDPG
from worlds import BASELINE_WORLD
from worlds import TURTLEBOT_WORLD

# prepare setting and values
np.set_printoptions(precision=3, suppress=True)

SEEDS = [0, 1, 2, 3, 4]
DDPGS = [IndividualDDPG,
         SharedExperienceDDPG,
         SharedNetworkDDPG,
         FederatedLearningDDPG]
PROGRESS_REWARD_FACTORS = [20, 40, 60, 80]
LEARNING_RATES = [0.1, 0.01, 0.001]

# launch roscore
uuid = roslaunch.rlutil.get_or_generate_uuid(options_runid=None, options_wait_for_master=False)
roslaunch.configure_logging(uuid)
roscore_launch = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_files=[], is_core=True)
roscore_launch.start()
# launch simulation
print('Simulation: Ready to start!')
uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
roslaunch.configure_logging(uuid)
world_launch = roslaunch.parent.ROSLaunchParent(uuid, ['/home/users/jpikman/catkin_ws/src/fl4sr/launch/frl_6.launch'])
world_launch.start()
time.sleep(5)

# run experiment
print('Simulation: Ready to run!')
for index in range(4):
    random.seed(3)
    np.random.seed(3)
    test = DDPGS[index](600, 256, TURTLEBOT_WORLD)
    success = False
    while not success:
        success, _, _ = test.run()
        if not success:
            # simulation failed, restart
            print('Simulation: Failed! Restaring...')
            time.sleep(5)
            # shutdown current simulation
            world_launch.shutdown()
            time.sleep(5)
            # launch new simulation
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)
            world_launch = roslaunch.parent.ROSLaunchParent(uuid, ['/home/users/jpikman/catkin_ws/src/fl4sr/launch/frl_6.launch'])
            world_launch.start()
            time.sleep(5)
            print('Simulation: Restart successful! Running...')
roscore_launch.shutdown()



