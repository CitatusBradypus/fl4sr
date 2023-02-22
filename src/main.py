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
# Commands are saved to command list, which are then then run.
# This is done so the experiment can be restarted after encoutering error.
COMMAND_LIST = [
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'IDDPG'],
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'SEDDPG'],
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'SNDDPG'],
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'IDDPG'],
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'FLDDPG', '--updatePeriod=2'],
    ['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'SwarmDDPG', '--updatePeriod=2']
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'PWDDPG', '--updatePeriod=2'],
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'RWDDPG', '--updatePeriod=2'],
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'MADDPG', '--updatePeriod=2'],
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'AllDDPG', '--updatePeriod=2'],
]

#COMMAND_LIST = []
#for eid in range(1, 8+1):
#    for nid in range(0, 4+1):
#        for wid in range(0, 4):
#            COMMAND_LIST += [['rosrun', 'fl4sr', 'experiment.py', '','IDDPG', '--worldNumber={}'.format(wid), 
#            '--pathActor=/home/users/jpikman/catkin_ws/src/fl4sr/src/data/RWDDPG-b-0.5-{}/weights/actor-final-{}.pkl'.format(eid, nid),
#            '--pathCritic=/home/users/jpikman/catkin_ws/src/fl4sr/src/data/RWDDPG-b-0.5-{}/weights/critic-final-{}.pkl'.format(eid, nid)]]
#for eid in range(1, 8+1):
#    for nid in range(0, 4+1):
#        for wid in range(0, 4):
#            COMMAND_LIST += [['rosrun', 'fl4sr', 'experiment.py', '','IDDPG', '--worldNumber={}'.format(wid),
#            '--pathActor=/home/users/jpikman/catkin_ws/src/fl4sr/src/data/MADDPG-h-b-0.5-{}/weights/actor-final-{}.pkl'.format(eid, nid),
#            '--pathCritic=/home/users/jpikman/catkin_ws/src/fl4sr/src/data/MADDPG-h-b-0.5-{}/weights/critic-final-{}.pkl'.format(eid, nid)]]
# for eid in range(4, 8+1):
#     for nid in range(0, 4+1):
#         for wid in range(0, 4):
#             COMMAND_LIST += [['rosrun', 'fl4sr', 'experiment.py', '','IDDPG', '--worldNumber={}'.format(wid),
#             '--pathActor=/home/users/jpikman/catkin_ws/src/fl4sr/src/data/FLDDPG-ms-{}/weights/actor-final-{}.pkl'.format(eid, nid),
#             '--pathCritic=/home/users/jpikman/catkin_ws/src/fl4sr/src/data/FLDDPG-ms-{}/weights/critic-final-{}.pkl'.format(eid, nid)]]

#COMMAND_LIST = 1 * COMMAND_LIST
#SEEDS = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
#for i in range(len(COMMAND_LIST)):
#    COMMAND_LIST[i] = COMMAND_LIST[i] + ['--seed', '{}'.format(SEEDS[(i//1)+2])]

#COMMAND_LIST = []
#for nid in range(4, 8+1):
#    for wid in range(0, 4):
#        COMMAND_LIST += [['rosrun', 'fl4sr', 'experiment.py', '','IDDPG', '--worldNumber={}'.format(wid),
#        '--pathActor=/home/users/jpikman/catkin_ws/src/fl4sr/src/data/PWDDPG-b-0.5-{}/weights/actor-final-4.pkl'.format(nid),
#        '--pathCritic=/home/users/jpikman/catkin_ws/src/fl4sr/src/data/PWDDPG-b-0.5-{}/weights/critic-final-4.pkl'.format(nid)]]

# PRINT
# Print all commands to before their execution.
for i in range(len(COMMAND_LIST)):    
    print(COMMAND_LIST[i])
#exit(0)

# RUN
# Execute each command until the success.
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

