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
from MomentumAveragingDDPG import MomentumAveragingDDPG
from AllDDPG import AllDDPG
from worlds import BASELINE_WORLD
from worlds import TURTLEBOT_WORLD_5
from worlds import TURTLEBOT_WORLD_6
from worlds import EVAL_WORLD_0
from worlds import EVAL_WORLD_1
from worlds import EVAL_WORLD_2
from worlds import EVAL_WORLD_3
from worlds import TURTLEBOT_WORLD_5_STARTS
from worlds import REAL_WORLD
from worlds import REAL_SIM_WORLD
from worlds import REAL_WORLD_8

# GLOBAL VARIABLES
DDPG = None
METHODS = {'IDDPG': IndividualDDPG,
           'SEDDPG': SharedExperienceDDPG,
           'SNDDPG': SharedNetworkDDPG,
           'FLDDPG': FederatedLearningDDPG,
           'PWDDPG': PositiveWeightingDDPG,
           'RWDDPG': RealWeightingDDPG,
           'MADDPG': MomentumAveragingDDPG,
           'AllDDPG': AllDDPG}

EPISODE_COUNT = 125
EPISODE_STEP_COUNT = 1024

LEARN_WORLD = REAL_WORLD_8

EVAL_WORLD = EVAL_WORLD_0

def experiment_learn(
        method: str,
        restart: bool,
        seed: int,
        update_step: int,
        update_period: int,
        reward_goal: float,
        reward_collision: float,
        reward_progress: float,
        reward_max_collision: float,
        factor_linear: float,
        factor_angular: float,
        discount_factor: float,
        is_progress: bool
    ) -> bool:
    """Run learning experiment with specified values.

    Returns:
        bool: If program finished correctly.
    """
    print(f"INSIDE experiment_learn | method: {method}, restart: {restart}, seed: {seed}, update_step: {update_step}, update_period: {update_period}, reward_goal: {reward_goal}, reward_collision: {reward_collision}, reward_progress: {reward_progress}, factor_linear: {factor_linear}, factor_angular: {factor_angular}, discount_factor: {discount_factor}, is_progress: {is_progress}")
    # ROS
    # launch roscore
    os.environ['ROS_MASTER_URI'] = f"http://192.168.210.127:11351/"
    #os.environ['GAZEBO_MASTER_URI'] = f"http://192.168.210.127:11371/"
    uuid = roslaunch.rlutil.get_or_generate_uuid(options_runid=None, options_wait_for_master=False)
    roslaunch.configure_logging(uuid)
    roscore_launch = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_files=[], is_core=True)
    roscore_launch.start()
    # launch simulation
    print('Simulation: Ready to start!')
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    world_launch = roslaunch.parent.ROSLaunchParent(uuid, [HOME + '/catkin_ws/src/fl4sr/launch/fl4sr_real_8.launch'])
    world_launch.start()
    time.sleep(5)
    # SETTINGS
    # set seeds
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    LEARN_ENV = 'Enviroment'
    # RUN
    print('Simulation: Ready to run!')
    if restart:
        with open('experiment.pickle', 'rb') as f:
            DDPG = pickle.load(f)
        DDPG.init_enviroment()
        print(f"INSIDE Enviroment | RESTART!")
    else:
        
        print(f"INSIDE Enviroment | INITIALISE DDPG")
        DDPG = METHODS[method](EPISODE_COUNT, EPISODE_STEP_COUNT, LEARN_WORLD, LEARN_ENV, reward_goal, reward_collision, reward_progress, reward_max_collision, factor_linear, factor_angular, discount_factor, method)
        if update_step is None and update_period is not None:
            DDPG.EPISODE_UPDATE = True
            DDPG.TIME_UPDATE = update_period
        elif update_period is not None:
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
    """Run evaluation experiment with specified values.

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
    world_launch = roslaunch.parent.ROSLaunchParent(uuid, [HOME + '/catkin_ws/src/fl4sr/launch/fl4sr_real_8.launch'])
    world_launch.start()
    time.sleep(5)
    # SETTINGS
    EPISODE_COUNT = 4
    robot_alives = 8
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
    elif world_number == 99:
        EVAL_WORLD = REAL_SIM_WORLD
    elif world_number == 100:
        EVAL_WORLD = REAL_WORLD_8
    # set env
    EVAL_ENV = 'Enviroment'
    # RUN
    print('Simulation: Ready to run!')
    if restart:
        with open('experiment.pickle', 'rb') as f:
            DDPG = pickle.load(f)
        DDPG.init_enviroment()
    else:
        DDPG = IndividualDDPG(EPISODE_COUNT, EPISODE_STEP_COUNT, EVAL_WORLD, EVAL_ENV,'EVAL-{}'.format(world_number))
        DDPG.agents_load(
            [path_actor for _ in range(robot_alives)],
            [path_critic for _ in range(robot_alives)]
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


def experiment_real(
        restart: bool,
        seed: int,
        world_number: int,
        path_actor: str,
        path_critic: str
    ) -> bool:
    """Run evaluation experiment with specified values.

    Returns:
        bool: If program finished correctly.
    """
    # SETTINGS
    EPISODE_COUNT = 5
    # set seeds
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    # set world
    EVAL_WORLD = REAL_WORLD
    # set env
    EVAL_ENV = 'RealEnviroment'
    # RUN
    print('Simulation: Ready to run!')
    if restart:
        with open('experiment.pickle', 'rb') as f:
            DDPG = pickle.load(f)
        DDPG.init_enviroment()
    else:
        DDPG = IndividualDDPG(EPISODE_COUNT, EPISODE_STEP_COUNT, EVAL_WORLD, EVAL_ENV,'REAL-{}'.format(world_number))
        DDPG.agents_load(
            [path_actor],
            [path_critic]
        ) 
    success, _, _ = DDPG.test_real()
    #roscore_launch.shutdown()
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
    # Experiments are defined by changing parameters in this file 
    # and also by setting arguments while starting running.

    # LOG SETTINGS
    np.set_printoptions(precision=3, suppress=True)

    # PARSE and their descriptions
    parser = argparse.ArgumentParser(
        description='Experiment script for fl4sr project.')
    parser.add_argument(
        '--mode',
        type=str,
        default='real',
        help='If method is supposted to learn, test in simulation or real-robot')
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
    parser.add_argument(
        '--reward_goal',
        type=float,
        help='Reward for reaching a goal.')
    parser.add_argument(
        '--reward_collision',
        type=float,
        help='Reward for collision.')
    parser.add_argument(
        '--reward_progress',
        type=float,
        help='Reward for the progress.')
    parser.add_argument(
        '--reward_max_collision',
        type=float,
        help='Reward for MAx collision dense.')
    parser.add_argument(
        '--factor_linear',
        type=float,
        help='Scaling factor for the linear velocity.')
    parser.add_argument(
        '--factor_angular',
        type=float,
        help='Scaling factor for the angular velocity.')
    parser.add_argument(
        '--discount_factor',
        type=float,
        help='discount_factor')   
    parser.add_argument(
        '--is_progress',
        type=bool,
        help='Determining the progress reward using step size or just using distances')
    parser.add_argument(
        'method', 
        type=str,
        help='Name of used method.')
    args = parser.parse_args()
    
    # ARGUMENTS
    assert args.method in METHODS, 'ERROR: Unknown method name.'
    
    # EXPERIMENT
    if args.mode == 'learn':
        print(f"It is in learning mode.")
        experiment_learn(args.method, args.restart, args.seed, args.updateStep, args.updatePeriod, args.reward_goal, args.reward_collision,\
                         args.reward_progress, args.reward_max_collision, args.factor_linear, args.factor_angular, args.discount_factor, args.is_progress)
    elif args.mode == 'real':
        print(f"It is in real mode")
        experiment_real(args.restart, args.seed, args.worldNumber, args.pathActor, args.pathCritic)
    elif args.mode == 'eval':
        print(f"It is in eval mode")
        experiment_test(args.restart, args.seed, args.worldNumber, args.pathActor, args.pathCritic)
    else: raise Exception('Wrong mode!')
