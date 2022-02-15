#! /usr/bin/env python3.8
import sys
import os
import time
import rospy
import pickle
import random
import argparse
import roslaunch
import numpy as np
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')
from IndividualDDPG import IndividualDDPG
from SharedNetworkDDPG import SharedNetworkDDPG
from SharedExperienceDDPG import SharedExperienceDDPG
from FederatedLearningDDPG import FederatedLearningDDPG
from worlds import BASELINE_WORLD
from worlds import TURTLEBOT_WORLD_5
from worlds import TURTLEBOT_WORLD_6


# GLOBAL VARIABLES
DDPG = None
METHODS = {'IDDPG': IndividualDDPG,
           'SEDDPG': SharedExperienceDDPG,
           'SNDDPG': SharedNetworkDDPG,
           'FLDDPG': FederatedLearningDDPG}
EPISODE_COUNT = 500
EPISODE_STEP_COUNT = 1024
WORLD = TURTLEBOT_WORLD_6

def experiment(
        method: str,
        restart: bool,
        seed: int
    ) -> bool:
    """Run experiment with specified values.

    Returns:
        bool: If program finished correctly.
    """
    # ROS
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
    # SETTINGS
    # set seeds
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    # RUN
    print('Simulation: Ready to run!')
    if restart:
        with open('experiment.pickle', 'rb') as f:
            DDPG = pickle.load(f)
        open('experiment.pickle', 'wb').close()
        DDPG.init_enviroment()
    else:
        DDPG = METHODS[args.method](EPISODE_COUNT, EPISODE_STEP_COUNT, WORLD)
    success, _, _ = DDPG.run()
    roscore_launch.shutdown()
    # RESULTS
    if not success:
        DDPG.terminate_enviroment()
        # save DDPG class
        with open('experiment.pickle', 'wb') as f:
            pickle.dump(DDPG, f)
        # write restart to file
        with open('main.info', 'w') as f:
            f.write('RESTART')
    return success

if __name__ == '__main__':
    # LOG SETTINGS
    np.set_printoptions(precision=3, suppress=True)
    # PARSE
    parser = argparse.ArgumentParser(
        description='Experiment script for fl4sr project.')
    parser.add_argument(
        'method', 
        type=str,
        help='Name of used method.')
    parser.add_argument(
        '--restart', 
        type=bool,
        help='Use saved class due to error.')
    parser.add_argument(
        '--seed',
        type=int,
        help='Seed for random generators.')
    args = parser.parse_args()
    # ARGUMENTS
    assert args.method in METHODS, 'ERROR: Unknown method name.'
    # EXPERIMENT
    experiment(args.method, args.restart, args.seed)

