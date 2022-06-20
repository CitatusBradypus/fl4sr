#! /usr/bin/env python3.8
import pickle
import sys
import os
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')
import matplotlib.pyplot as plt
import numpy as np
import time
path_data = HOME +'/catkin_ws/src/fl4sr/src/data'
path_name = 'FederatedLearningDDPG-20211130-235816'
rewards = np.load(path_data + '/'+ path_name + '/log/rewards-final.npy')
succeded = np.load(path_data + '/' + path_name + '/log/succeded-final.npy')
with open(path_data + '/' + path_name + '/log/parameters.pkl', 'rb') as f:
    parameters = pickle.load(f)

for key in parameters:
    print('{} = {}'.format(key, parameters[key]))

xs = np.arange(len(rewards))
ys_avg_rewards = np.mean(rewards, axis=1)
ys_std_rewards = np.std(rewards, axis=1)
ys_epsilon = (0 * xs)
ys_succ_counter_1d = np.sum(succeded.astype(float), axis=1)
print(ys_succ_counter_1d.shape)
print(ys_succ_counter_1d)
ys_succ_counter = np.sum(ys_succ_counter_1d)
print(ys_succ_counter)

plt.plot(xs, ys_avg_rewards)
plt.fill_between(xs, ys_avg_rewards - ys_std_rewards, ys_avg_rewards + ys_std_rewards, alpha=0.5)
plt.plot(xs, ys_epsilon)
plt.show()
