#! /usr/bin/env python3.8
import sys
import os
import time
import rospy
import random
import subprocess
import numpy as np


list_dir = os.listdir('eval_model')
cur_path = os.path.join(os.getcwd(), 'eval_model')
print(list_dir)
list_dir_results = [d for d in list_dir if not 'ipynb' in d]
print(list_dir_results)

# COMMANDS
# Commands are saved to command list, which are then then run.
# This is done so the experiment can be restarted after encoutering error.

COMMAND_LIST = []
for dir_result in list_dir_results:
    path_dir_setting = os.path.join(cur_path, dir_result)
    list_dir_setting = os.listdir(path_dir_setting)
    print(list_dir_setting)
    for setting in list_dir_setting:
        path_algo = os.path.join(path_dir_setting, setting)
        list_algo = os.listdir(path_algo)
        for algo in list_algo:
            path_model = os.path.join(path_algo, algo)
            list_model = os.listdir(path_model)
            name_critic = [critic for critic in list_model if 'critic' in critic][0]
            name_actor = [actor for actor in list_model if 'actor' in actor][0]
            model_name = f'{dir_result}-{setting}-{algo}'
            print("wow")
            path_critic = os.path.join(path_algo, name_critic)
            path_actor = os.path.join(path_algo, name_actor)
            COMMAND_LIST.append(['rosrun', 'fl4sr', 'experiment.py', f'--mode={"eval"}', 'IDDPG',
            f'--restart={False}',
            f'--seed={101}',
            f'--worldNumber={101}',
            f'--model_name={model_name}',
            f'--env_name={"Enviroment_eval"}',
            f'--pathActor={path_actor}',
            f'--pathCritic={path_critic}'])



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

