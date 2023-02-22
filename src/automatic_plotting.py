import numpy as np
import os
import csv
import matplotlib.pyplot as plt

def get_path_log(path_data: str, 
    exp_name: str
    ) -> str:
    path_exp = os.path.join(path_data, exp_name)
    path_log = os.path.join(path_exp, 'log')
    return path_log

def get_path_reward(path_data: str, 
    exp_name: str
    ) -> str:
    print(f"path_data: {path_data}")
    path_exp = os.path.join(path_data, exp_name)
    print(f"path_exp: {path_exp}")
    path_log = os.path.join(path_exp, 'log')
    path_reward = os.path.join(path_log, 'rewards.npy')
    print(f"path_reward: {path_reward}")
    return path_reward

def plot_reward(data_reward: np.array,
    path_save: str,
    exp_name: str,
    ) -> None:
    arr_rewards_T = data_reward.T
    values = arr_rewards_T
    print(f"shape of values: {values.shape}")
    
    num_time_steps = values.shape[1]
    list_time_steps = [i for i in range(num_time_steps)]
    num_agents = 4
    list_agents = [f"Robot {i}" for i in range(num_agents)]
    
    figure, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
    #plt.xlabel('Episodes')
    #plt.ylabel('Rewards')
    ax.plot(list_time_steps, values.T)
    ax.set_title(f'Experiment name: {exp_name}')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Rewards')
    ax.set_ylim(0, 15)
    ax.legend(list_agents)

    plot_name = exp_name + '.png'
    path_plot = os.path.join(path_save, plot_name)
    figure.savefig(path_plot)
    return


if __name__=="__main__":
    HOME = os.environ['HOME']
    path_data = HOME + '/catkin_ws/src/fl4sr/src/data'

    csv_files = open("plot_reward_exps.csv", 'r')
    list_exp = list(csv.reader(csv_files, delimiter=","))
    
    for exp_name in list_exp[0]:
        path_log = get_path_log(path_data, exp_name)
        path_reward = get_path_reward(path_data, exp_name)
        is_path_exist = os.path.exists(path_reward)
        if is_path_exist == True:
            print(f"data is found at: {path_reward}")
            data_reward = np.load(path_reward)
            plot_reward(data_reward, path_log, exp_name)
        else: print(f"No data is found at: {path_reward}")
    csv_files.close()
    print(f"Plots are successfully generated!")

    

