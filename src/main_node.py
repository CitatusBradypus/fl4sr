#! /usr/bin/env python3.8
import sys
import os
import multiprocessing as mp
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


class MainWorker(mp.Process):

    def __init__(self, num_port):
        super(MainWorker, self).__init__()
        self.num_port = num_port
        self.ROS_MASTER_URI = f"http://localhost:1135{self.num_port}/"
        self.GAZEBO_MASTER_URI = f"http://localhost:1137{self.num_port}/"
        
        self.dict_hypotheses = {'hp1': {'reward_goal': [100.0, 50.0], 'reward_collision': [-10.0, -100.0], 'reward_progress': [40.0, 30.0]},\
                           'hp2': {'factor_angular': [1.0, 1.5]},\
                           'hp3': {'algorithms': ['IDDPG', 'SEDDPG', 'SNDDPG', 'FLDDPG']}}
        self.COMMAND_LIST = []


    def set_command(self):
        #dict_keys = list(self.dict_hypotheses.keys())
        # list_all_keys = []
        # for value in self.dict_hypothese.values():
        #     list_keys = []
        #     for key in zip(value):
        #         list_keys.append(key)
        #     list_all_keys.append(list_keys)

            
        for rg, rc, rp in zip(self.dict_hypotheses['hp1']['reward_goal'], self.dict_hypotheses['hp1']['reward_collision'], self.dict_hypotheses['hp1']['reward_progress']):
            for fa in self.dict_hypotheses['hp2']['factor_angular']:
                for algo in self.dict_hypotheses['hp3']['algorithms']:
                    self.COMMAND_LIST.append(['rosrun', 'fl4sr', 'experiment.py', f'--mode={"learn"}', f'--reward_goal={rg}',\
                    f'--reward_collision={rc}',f'--reward_progress={rp}', '--factor_linear=0.25',f'--factor_angular={fa}', algo])
    def set_env_variables(self):
        os.environ['ROS_MASTER_URI'] = self.ROS_MASTER_URI
        os.environ['GAZEBO_MASTER_URI'] = self.GAZEBO_MASTER_URI
    
    def print_command(self):
        for i in range(len(COMMAND_LIST)):    
            print(COMMAND_LIST[i])
            
    def run_command(self):
        for command in self.COMMAND_LIST:
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
    def run(self):
        self.set_env_variables()
        self.set_command()
        self.run_command()
        
        



        # dict_reward = {'reward_goal': [100.0, 50.0], 'reward_collision': [-10.0, -100.0], 'reward_progress': [40.0, 30.0]}
        # dict_factor = {'factor_angular': [1.0, 1.5]}
        # dict_algorithms = {'algorithms': ['IDDPG', 'SEDDPG', 'SNDDPG', 'FLDDPG']}

        # COMMAND_LIST = []
        # for rg, rc, rp in zip(dict_reward['reward_goal'], dict_reward['reward_collision'], dict_reward['reward_progress']):
        #     for fa in dict_factor['factor_angular']:
        #         for algo in dict_algorithms['algorithms']:
        #             COMMAND_LIST.append(['rosrun', 'fl4sr', 'experiment.py', f'--mode={"learn"}', f'--reward_goal={rg}',\
        #             f'--reward_collision={rc}',f'--reward_progress={rp}', '--factor_linear=0.25',f'--factor_angular={fa}', algo])


# COMMAND_LIST = [
#     ['rosrun', 'fl4sr', 'experiment.py', f'--mode={"learn"}', '--reward_goal=100.0', '--reward_collision=-10.0','--reward_progress=40.0', '--factor_linear=0.25','--factor_angular=1.0', 'IDDPG'],
#     ['rosrun', 'fl4sr', 'experiment.py', f'--mode={"learn"}',  '--reward_goal=100.0', '--reward_collision=-10.0','--reward_progress=40.0', '--factor_linear=0.25','--factor_angular=1.0', 'SEDDPG'],
#     ['rosrun', 'fl4sr', 'experiment.py', f'--mode={"learn"}',  '--reward_goal=100.0', '--reward_collision=-10.0','--reward_progress=40.0', '--factor_linear=0.25','--factor_angular=1.0', 'SNDDPG'],
#     ['rosrun', 'fl4sr', 'experiment.py', f'--mode={"learn"}',  '--updatePeriod=2', '--reward_goal=100.0', '--reward_collision=-10.0','--reward_progress=40.0', '--factor_linear=0.25','--factor_angular=1.0', 'FLDDPG'],
#     ['rosrun', 'fl4sr', 'experiment.py', f'--mode={"learn"}', '--reward_goal=50.0', '--reward_collision=-100.0','--reward_progress=30.0', '--factor_linear=0.25','--factor_angular=1.5', 'IDDPG'],
#     ['rosrun', 'fl4sr', 'experiment.py', f'--mode={"learn"}',  '--reward_goal=50.0', '--reward_collision=-100.0','--reward_progress=30.0', '--factor_linear=0.25','--factor_angular=1.5', 'SEDDPG'],
#     ['rosrun', 'fl4sr', 'experiment.py', f'--mode={"learn"}',  '--reward_goal=50.0', '--reward_collision=-100.0','--reward_progress=30.0', '--factor_linear=0.25','--factor_angular=1.5', 'SNDDPG'],
#     ['rosrun', 'fl4sr', 'experiment.py', f'--mode={"learn"}',  '--updatePeriod=2', '--reward_goal=50.0', '--reward_collision=-100.0','--reward_progress=30.0', '--factor_linear=0.25','--factor_angular=1.5', 'FLDDPG'],
#     ['rosrun', 'fl4sr', 'experiment.py', f'--mode={"learn"}', '--reward_goal=100.0', '--reward_collision=-10.0','--reward_progress=40.0', '--factor_linear=0.25','--factor_angular=1.5', 'IDDPG'],
#     ['rosrun', 'fl4sr', 'experiment.py', f'--mode={"learn"}',  '--reward_goal=100.0', '--reward_collision=-10.0','--reward_progress=40.0', '--factor_linear=0.25','--factor_angular=1.5', 'SEDDPG'],
#     ['rosrun', 'fl4sr', 'experiment.py', f'--mode={"learn"}',  '--reward_goal=100.0', '--reward_collision=-10.0','--reward_progress=40.0', '--factor_linear=0.25','--factor_angular=1.5', 'SNDDPG'],
#     ['rosrun', 'fl4sr', 'experiment.py', f'--mode={"learn"}',  '--updatePeriod=2', '--reward_goal=100.0', '--reward_collision=-10.0','--reward_progress=40.0', '--factor_linear=0.25','--factor_angular=1.0', 'FLDDPG'],
#     ['rosrun', 'fl4sr', 'experiment.py', f'--mode={"learn"}', '--reward_goal=50.0', '--reward_collision=-100.0','--reward_progress=30.0', '--factor_linear=0.25','--factor_angular=1.0', 'IDDPG'],
#     ['rosrun', 'fl4sr', 'experiment.py', f'--mode={"learn"}',  '--reward_goal=50.0', '--reward_collision=-100.0','--reward_progress=30.0', '--factor_linear=0.25','--factor_angular=1.0', 'SEDDPG'],
#     ['rosrun', 'fl4sr', 'experiment.py', f'--mode={"learn"}',  '--reward_goal=50.0', '--reward_collision=-100.0','--reward_progress=30.0', '--factor_linear=0.25','--factor_angular=1.0', 'SNDDPG'],
#     ['rosrun', 'fl4sr', 'experiment.py', f'--mode={"learn"}',  '--updatePeriod=2', '--reward_goal=50.0', '--reward_collision=-100.0','--reward_progress=30.0', '--factor_linear=0.25','--factor_angular=1.0', 'FLDDPG'],
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'PWDDPG', '--updatePeriod=2'],
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'RWDDPG', '--updatePeriod=2'],
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'MADDPG', '--updatePeriod=2'],
    #['rosrun', 'fl4sr', 'experiment.py', 'learn=True', 'AllDDPG', '--updatePeriod=2'],
#]

# COMMAND_LIST = []
# #for eid in range(1, 8+1):
# #    for nid in range(0, 4+1):
# #        for wid in range(0, 4):
# #            COMMAND_LIST += [['rosrun', 'fl4sr', 'experiment.py', '','IDDPG', '--worldNumber={}'.format(wid), 
# #            '--pathActor=/home/users/jpikman/catkin_ws/src/fl4sr/src/data/RWDDPG-b-0.5-{}/weights/actor-final-{}.pkl'.format(eid, nid),
# #            '--pathCritic=/home/users/jpikman/catkin_ws/src/fl4sr/src/data/RWDDPG-b-0.5-{}/weights/critic-final-{}.pkl'.format(eid, nid)]]
# #for eid in range(1, 8+1):
# #    for nid in range(0, 4+1):
# #        for wid in range(0, 4):
# #            COMMAND_LIST += [['rosrun', 'fl4sr', 'experiment.py', '','IDDPG', '--worldNumber={}'.format(wid),
# #            '--pathActor=/home/users/jpikman/catkin_ws/src/fl4sr/src/data/MADDPG-h-b-0.5-{}/weights/actor-final-{}.pkl'.format(eid, nid),
# #            '--pathCritic=/home/users/jpikman/catkin_ws/src/fl4sr/src/data/MADDPG-h-b-0.5-{}/weights/critic-final-{}.pkl'.format(eid, nid)]]
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
# for i in range(len(COMMAND_LIST)):    
#     print(COMMAND_LIST[i])
#exit(0)

# RUN
# Execute each command until the success.
# for command in COMMAND_LIST:
#     success = False
#     restart_command = []
#     while not success:
#         print(command + restart_command)
#         subprocess.run(command + restart_command)

#         with open('main.info', 'r') as f:
#             result = f.readline()
#         open('main.info', 'w').close()
#         if result == '':
#             print('COMMAND OK')
#             success = True
#         else:
#             restart_command = ['--restart', 'True']

