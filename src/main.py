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


COMMAND_LIST = [
    ['rosrun', 'fl4sr', 
     'experiment.py', 
     'IDDPG'],
    ['rosrun', 'fl4sr', 
     'experiment.py', 
     'SEDDPG'],
    ['rosrun', 'fl4sr',
     'experiment.py',
     'SNDDPG'],
    ['rosrun', 'fl4sr',
     'experiment.py',
     'FLDDPG']
]

for command in COMMAND_LIST:
    success = False
    restart_command = []
    while not success:
        subprocess.run(command + restart_command)

        with open('main.info', 'r') as f:
            result = f.readline()
        open('main.info', 'w').close()
        if result == '':
            print('COMMAND OK')
            success = True
        else:
            restart_command = ['--restart', 'True']
