#!/usr/bin/python3.8
import sys
import os
import time
import rospy
import random
import subprocess
import numpy as np
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')


COMMAND_LIST = [
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'IDDPG'],
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'SEDDPG'],
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'SNDDPG'],
    #['rosrun', 'fl4sr', 'experiment.py', f'--mode={"real"}', 'IDDPG', f'--pathActor={HOME}/catkin_ws/src/fl4sr/src/data/real_robot/IDDPG/actor-final-0.pkl', f'--pathCritic={HOME}/catkin_ws/src/fl4sr/src/data/real_robot/IDDPG/critic-final-0.pkl'],
    #['rosrun', 'fl4sr', 'experiment.py', f'--mode={"real"}', 'SNDDPG', f'--pathActor={HOME}/catkin_ws/src/fl4sr/src/data/real_robot/SNDDPG/actor-final-0.pkl', f'--pathCritic={HOME}/catkin_ws/src/fl4sr/src/data/real_robot/SNDDPG/critic-final-0.pkl'],
    #['rosrun', 'fl4sr', 'experiment.py', f'--mode={"real"}', 'SEDDPG', f'--pathActor={HOME}/catkin_ws/src/fl4sr/src/data/real_robot/SEDDPG/actor-final-4.pkl', f'--pathCritic={HOME}/catkin_ws/src/fl4sr/src/data/real_robot/SEDDPG/critic-final-4.pkl'],
    
    #['rosrun', 'fl4sr', 'experiment.py', f'--mode={"eval"}', 'IDDPG', f'--worldNumber={99}', f'--pathActor={HOME}/catkin_ws/src/fl4sr/src/data/real_robot/IDDPG/actor-final-0.pkl', f'--pathCritic={HOME}/catkin_ws/src/fl4sr/src/data/real_robot/IDDPG/critic-final-0.pkl'],
    #['rosrun', 'fl4sr', 'experiment.py', f'--mode={"eval"}', 'SNDDPG', f'--worldNumber={99}', f'--pathActor={HOME}/catkin_ws/src/fl4sr/src/data/real_robot/SNDDPG/actor-final-0.pkl', f'--pathCritic={HOME}/catkin_ws/src/fl4sr/src/data/real_robot/SNDDPG/critic-final-0.pkl'],
    ['rosrun', 'fl4sr', 'experiment.py', f'--mode={"eval"}', 'SEDDPG', f'--worldNumber={99}', f'--pathActor={HOME}/catkin_ws/src/fl4sr/src/data/real_robot/SEDDPG/actor-final-4.pkl', f'--pathCritic={HOME}/catkin_ws/src/fl4sr/src/data/real_robot/SEDDPG/critic-final-4.pkl'],
    #['rosrun', 'fl4sr', 'experiment.py', f'--mode={"eval"}', 'FLDDPG', f'--worldNumber={99}', f'--pathActor={HOME}/catkin_ws/src/fl4sr/src/data/real_robot/FLDDPG/actor-final-4.pkl', f'--pathCritic={HOME}/catkin_ws/src/fl4sr/src/data/real_robot/FLDDPG/critic-final-4.pkl'],
    
    #['rosrun', 'fl4sr', 'experiment.py', f'--mode={"real"}', 'FLDDPG', f'--pathActor={HOME}/catkin_ws/src/fl4sr/src/data/real_robot/FLDDPG/actor-final-4.pkl', f'--pathCritic={HOME}/catkin_ws/src/fl4sr/src/data/real_robot/FLDDPG/critic-final-4.pkl'],
    
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'PWDDPG', '--updatePeriod=2'],
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'RWDDPG', '--updatePeriod=2'],
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'MADDPG', '--updatePeriod=2'],
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'AllDDPG', '--updatePeriod=2'],
]


for i in range(len(COMMAND_LIST)):    
    print(COMMAND_LIST[i])


for command in COMMAND_LIST:
    print(f"Experiment Started!: {command}")
    success = False
    restart_command = []
    while not success:
        print(command + restart_command)
        subprocess.run(command + restart_command)

        with open('main.info', 'r') as f:
            result = f.readline()
        open('main.info', 'w').close()
        if result == '':
            print('COMMAND OK')
            success = True
        else:
            restart_command = ['--restart', 'True']
