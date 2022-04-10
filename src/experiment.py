#! /usr/bin/env python3.8
import string
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
from PositiveWeightingDDPG import PositiveWeightingDDPG
from RealWeightingDDPG import RealWeightingDDPG
from worlds import BASELINE_WORLD
from worlds import TURTLEBOT_WORLD_5
from worlds import TURTLEBOT_WORLD_6
from worlds import EVAL_WORLD_0
from worlds import EVAL_WORLD_1
from worlds import EVAL_WORLD_2
from worlds import EVAL_WORLD_3


# GLOBAL VARIABLES
DDPG = None
METHODS = {'IDDPG': IndividualDDPG,
           'SEDDPG': SharedExperienceDDPG,
           'SNDDPG': SharedNetworkDDPG,
           'FLDDPG': FederatedLearningDDPG,
           'PWDDPG': PositiveWeightingDDPG,
           'RWDDPG': RealWeightingDDPG}

EPISODE_COUNT = 125
EPISODE_STEP_COUNT = 1024

LEARN_WORLD = TURTLEBOT_WORLD_5

EVAL_WORLD = EVAL_WORLD_0

def experiment_learn(
        method: str,
        restart: bool,
        seed: int,
        update_step: int,
        update_period: int,
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
        DDPG.init_enviroment()
    else:
        DDPG = METHODS[method](EPISODE_COUNT, EPISODE_STEP_COUNT, LEARN_WORLD)
        if update_step is None and update_period is not None:
            DDPG.EPISODE_UPDATE = True
            DDPG.TIME_UPDATE = update_period
        else:
            DDPG.EPISODE_UPDATE = False
            DDPG.TIME_UPDATE = update_period
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


def experiment_test(
        restart: bool,
        seed: int,
        world_number: int,
        path_actor: str,
        path_critic: str
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
    world_launch = roslaunch.parent.ROSLaunchParent(uuid, ['/home/users/jpikman/catkin_ws/src/fl4sr/launch/frl_eval_easy.launch'])
    world_launch.start()
    time.sleep(5)
    # SETTINGS
    # set seeds
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    # set world
    if world_number == 0:
        EVAL_WORLD = EVAL_WORLD_0
    elif world_number == 1:
        EVAL_WORLD = EVAL_WORLD_1
    elif world_number == 2:
        EVAL_WORLD = EVAL_WORLD_2
    elif world_number == 3:
        EVAL_WORLD = EVAL_WORLD_3
    # RUN
    print('Simulation: Ready to run!')
    if restart:
        with open('experiment.pickle', 'rb') as f:
            DDPG = pickle.load(f)
        DDPG.init_enviroment()
    else:
        DDPG = IndividualDDPG(EPISODE_COUNT, EPISODE_STEP_COUNT, EVAL_WORLD)
        DDPG.agents_load(
            [path_actor],
            [path_critic]
        )
    success, _, _ = DDPG.test()
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
        'learn',
        type=bool,
        help='If method is supposted to learn or test')
    parser.add_argument(
        '--worldNumber', 
        type=int,
        help='Specifier for world type.')
    parser.add_argument(
        '--pathActor', 
        type=str,
        help='Path to actor.')
    parser.add_argument(
        '--pathCritic', 
        type=str,
        help='Path to critic.')
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
    parser.add_argument(
        '--updateStep',
        type=bool,
        help='If the federated learning is done in episodes.')
    parser.add_argument(
        '--updatePeriod',
        type=int,
        help='Period of federated update.')
    args = parser.parse_args()
    # ARGUMENTS
    assert args.method in METHODS, 'ERROR: Unknown method name.'
    # EXPERIMENT
    if args.learn:
        experiment_learn(args.method, args.restart, args.seed, args.updateStep, args.updatePeriod)
    else:
        experiment_test(args.restart, args.seed, args.worldNumber, args.pathActor, args.pathCritic)
