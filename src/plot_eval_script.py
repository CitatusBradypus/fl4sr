#!/usr/bin/python3.8
'''
1. Match Evaluation exp folders with training exp folders
2. Plot the result of each experiment
3. Save the plots in the training exp folders.
'''

import numpy as np
import argparse
import os
import math
import matplotlib.pyplot as plt

HOME = os.environ['HOME']
def compute_avg_std_quartiles(list_time):
    list_non_zero_time = []
    for time in list_time:
        if time > 0: 
            list_non_zero_time.append(time)
    
    array_non_zero_time = np.array(list_non_zero_time) 
    avg_non_zero_time = np.average(array_non_zero_time)
    std_non_zero_time = np.std(array_non_zero_time)
    #q1_non_zero_time = np.quantile(array_non_zero_time, 0.25)
    #q2_non_zero_time = np.quantile(array_non_zero_time, 0.5)
    #q3_non_zero_time = np.quantile(array_non_zero_time, 0.75)    
    if math.isnan(float(avg_non_zero_time)):
        avg_non_zero_time = 0.0
    return [avg_non_zero_time, std_non_zero_time] #,[q1_non_zero_time, q2_non_zero_time, q3_non_zero_time]]

if __name__=="__main__":

    # PARSE and their descriptions
    parser = argparse.ArgumentParser(
        description='Experiment script for fl4sr project.')
    parser.add_argument(
        '--name_exp_dir',
        type=str,
        help='a name of training exp directory containing all the experiment subdirs')
    parser.add_argument(
        '--name_eval_dir',
        type=str,
        help='a name of training eval directory containing all the experiment subdirs')

    # Read path of exp and eval dirs
    args = parser.parse_args()
    path_data = HOME + '/catkin_ws/src/fl4sr/src/data'

    path_exp = os.path.join(path_data, args.name_exp_dir)
    list_exp = os.listdir(path_exp)
    list_exp = [dir_name for dir_name in list_exp if not 'png' in dir_name]
    list_exp = [dir_name for dir_name in list_exp if len(dir_name) > 13]

    path_eval = os.path.join(path_data, args.name_eval_dir)
    
    list_eval = os.listdir(path_eval)
    list_exp = list_exp[0:len(list_eval)-1]

    # Map between name_exp and name_eval
    dict_exp_eval = dict([[name_exp, name_eval] for name_exp, name_eval in zip(list_exp, list_eval)])
    dict_eval_exp = dict([[name_eval, name_exp] for name_exp, name_eval in zip(list_exp, list_eval)])
    print(dict_eval_exp)
    num_robots = 4 
    for name_eval in list_eval:

        # Read list of 3 set of results.
        path_name_eval = os.path.join(path_eval, name_eval)
        path_eval_log = os.path.join(path_name_eval, 'log')
        path_robot_succeeded = os.path.join(path_eval_log, 'list_robot_succeeded.npy')
        path_arrival_time = os.path.join(path_eval_log, 'list_arrival_time.npy')
        path_traj_eff = os.path.join(path_eval_log, 'list_traj_eff.npy')
        list_robot_succeeded = np.load(path_robot_succeeded)
        list_arrival_time = np.load(path_arrival_time)
        list_traj_eff = np.load(path_traj_eff)

        robot_succeeded_num = [[sum(agent)/len(agent) for agent in np.array(list_robot_succeeded).T]]
        avg_std_quartiles_arrival_time = [compute_avg_std_quartiles([time for time in agent if time >0]) for agent in np.array(list_arrival_time).T]
        avg_std_quartiles_traj_eff = [compute_avg_std_quartiles([traj for traj in agent if traj >0]) for agent in np.array(list_traj_eff).T]

        avg_arrival_time = []
        std_arrival_time = []
        #quartiles_arrival_time = []
        avg_traj_eff = []
        std_traj_eff = []
        #quartiles_traj_eff = []

        for i in range(num_robots):
            avg_arrival_time.append(avg_std_quartiles_arrival_time[i][0])
            std_arrival_time.append(avg_std_quartiles_arrival_time[i][1])
            #quartiles_arrival_time.append(avg_std_quartiles_arrival_time[i][2])

            avg_traj_eff.append(avg_std_quartiles_traj_eff[i][0])
            std_traj_eff.append(avg_std_quartiles_traj_eff[i][1])
            #quartiles_traj_eff.append(avg_std_quartiles_traj_eff[i][2])

        avg_avg_arrival_time = np.average(avg_arrival_time)
        std_avg_arrival_time = np.std(avg_arrival_time)
        avg_avg_traj_eff = np.average(avg_traj_eff)
        std_avg_traj_eff = np.std(avg_traj_eff)


        avg_robot_succeeded = np.average(robot_succeeded_num)

        robot_succeeded_num[0].append(avg_robot_succeeded)
        avg_arrival_time.append(avg_avg_arrival_time)
        std_arrival_time.append(std_avg_arrival_time)
        avg_traj_eff.append(avg_traj_eff)
        std_traj_eff.append(std_traj_eff)

        # Plot the results
        figure, axs = plt.subplots(1, 3, figsize=(25,10))
        x = ['agent'+str(i) for i in range(4)]
        x.append('Average Agent')

        axs[0].bar(x, robot_succeeded_num[0])
        axs[0].set_ylim(0,1)
        axs[1].bar(x, avg_arrival_time)
        #axs[1][i].set_ylim()
        axs[2].bar(x, avg_avg_traj_eff)

        axs[0].set_xlabel('ID of Robots')
        axs[0].set_ylabel('Success Rate')
        axs[1].set_xlabel('ID of Robots')
        axs[1].set_ylabel('Arrival Time (s)')
        axs[2].set_xlabel('ID of Robots')
        axs[2].set_ylabel('Trajectory Efficiency')
        name_exp = dict_eval_exp[name_eval]
        figure.suptitle(f'{name_exp}')

        # Save the plot
        path_name_exp = os.path.join(path_exp, name_exp)
        path_eval_plot = os.path.join(path_name_exp, 'eval_plot.png')
        figure.savefig(path_eval_plot)
        print(f'plot is successfully saved!')











    
    


    

    