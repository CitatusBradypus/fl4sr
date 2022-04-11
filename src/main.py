#! /usr/bin/env python3.8
import sys
import os
import time
import rospy
import random
import subprocess
import numpy as np
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')

# COMMANDS
COMMAND_LIST = [
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'IDDPG'],
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'SEDDPG'],
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'SNDDPG'],
    ['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'FLDDPG', '--updatePeriod=2'],
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'PWDDPG', '--updatePeriod=2'],
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'RWDDPG', '--updatePeriod=2'],
]

#COMMAND_LIST = []
#for nid in range(1, 8+1):
#    for wid in range(0, 4):
#        COMMAND_LIST += [['rosrun', 'fl4sr', 'experiment.py', '','IDDPG', '--worldNumber={}'.format(wid), 
#        '--pathActor=/home/users/jpikman/catkin_ws/src/fl4sr/src/data/FLDDPG-s-p-2-{}/weights/actor-final-4.pkl'.format(nid),
#        '--pathCritic=/home/users/jpikman/catkin_ws/src/fl4sr/src/data/FLDDPG-s-p-2-{}/weights/critic-final-4.pkl'.format(nid)]]

SEEDS = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
COMMAND_LIST = 1 * COMMAND_LIST
for i in range(len(COMMAND_LIST)):
    COMMAND_LIST[i] = COMMAND_LIST[i] + ['--seed', '{}'.format(SEEDS[(i//2)+0])]
#COMMAND_LIST = [['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'RWDDPG', '--updatePeriod=2', '-seed=14']] + COMMAND_LIST

# PRINT 
for i in range(len(COMMAND_LIST)):    
    print(COMMAND_LIST[i])
#exit(0)

# RUN
for command in COMMAND_LIST:
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

