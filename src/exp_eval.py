#! /usr/bin/env python3.8
import numpy as np
import sys
import os
import csv
import time
import rospy
import random
import matplotlib.pyplot as plt
import argparse
import subprocess

HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')



def get_path_log(path_data: str, 
    exp_name: str
    ) -> str:
    path_exp = os.path.join(path_data, exp_name)
    path_log = os.path.join(path_exp, 'log')
    return path_log

def get_path_weights(path_data: str, 
    exp_name: str
    ) -> str:
    path_exp = os.path.join(path_data, exp_name)
    path_weights = os.path.join(path_exp, 'weights')
    return path_weights


def extract_key_value(list_weights: list
    ) -> dict:

    list_key_0 = []
    list_key_1 = []
    list_key_2 = []    
    for name_weights in list_weights:
        list_str = name_weights.split('-')
        print(f"list_str: {list_str}")
        list_str[2] = list_str[2][0]
        list_key_0.append(list_str[0])
        list_key_1.append(list_str[1])
        list_key_2.append(list_str[2])
    
    list_key_0 = list(dict.fromkeys(list_key_0))
    list_key_1 = list(dict.fromkeys(list_key_1))
    list_key_2 = list(dict.fromkeys(list_key_2))

    dict_key_value = {'neural_network': list_key_0, 'num_episodes': list_key_1,
                    'agent_ids': list_key_2}

    return dict_key_value

        
        
        

if __name__=="__main__":

    # PARSE and their descriptions
    parser = argparse.ArgumentParser(
        description='Experiment script for fl4sr project.')
    parser.add_argument(
        '--name_parent_dir',
        type=str,
        help='a name of parent directory containing all the experiment subdirs')
    args = parser.parse_args()
    path_data = HOME + '/catkin_ws/src/fl4sr/src/data'
    dir_name = args.name_parent_dir
    path_parent = os.path.join(path_data, dir_name)
    list_exp = os.listdir(path_parent)



    
    COMMAND_LIST = []
    # Using a directory to read the file names.
    for exp_name in list_exp:
        path_weights = get_path_weights(path_parent, exp_name)
        is_path_exist = os.path.exists(path_weights)
        if is_path_exist == True:
            list_weights = os.listdir(path_weights)
            dict_key_value = extract_key_value(list_weights)
            for agent_id in dict_key_value['agent_ids']:
                #for episode in dict_key_value['num_episodes']:
                ##### RUN EVALUATION WITH GIVEN (1) AGENT_ID, (2) EPISODE #####
                # run_evaluation()
                actor_model_name = f"actor-{'final'}-{agent_id}.pkl"
                critic_model_name = f"critic-{'final'}-{agent_id}.pkl"
                path_actor = os.path.join(path_weights, actor_model_name)
                path_critic = os.path.join(path_weights, critic_model_name)
                
                exp_name_agent = exp_name + '_' + str(agent_id)
                COMMAND_LIST.append(['rosrun', 'fl4sr', 'experiment_limit.py', f'--mode={"eval"}', 'IDDPG',
                                    f'--restart={False}',
                                    f'--seed={101}',
                                    f'--worldNumber={101}',
                                    f'--model_name={exp_name_agent}',
                                    f'--env_name={"Enviroment_eval"}',
                                    f'--pathActor={path_actor}',
                                    f'--pathCritic={path_critic}'])

        else: print(f"No data is found at: {path_weights}")
    
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
            print(f"result: {result}")
            if result == '':
                print('COMMAND OK')
                success = True
            else:
                restart_command = ['--restart', 'False']
