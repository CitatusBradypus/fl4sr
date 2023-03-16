import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import argparse

def get_path_log(path_data: str, 
    exp_name: str
    ) -> str:
    path_exp = os.path.join(path_data, exp_name)
    path_log = os.path.join(path_exp, 'log')
    return path_log

def get_path_reward(path_data: str, 
    exp_name: str
    ) -> str:
    path_exp = os.path.join(path_data, exp_name)
    path_log = os.path.join(path_exp, 'log')
    path_reward = os.path.join(path_log, 'rewards.npy')
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
    ax.set_ylim(-5, 5)
    ax.legend(list_agents)

    plot_name = exp_name + '.png'
    path_plot = os.path.join(path_save, plot_name)
    figure.savefig(path_plot)
    return

def plot_reward_algo_up(avg_reward: np.array,
    std_reward: np.array,
    path_save: str,
    exp_name: str,
    ) -> None:
    arr_rewards_T = avg_reward.T
    std_rewards_T = std_reward.T
    values = arr_rewards_T
    values_std = std_rewards_T
    
    num_time_steps = values.shape[1]
    list_time_steps = [i for i in range(num_time_steps)]
    num_agents = 4
    list_agents = [f"Robot {i}" for i in range(num_agents)]
    
    figure, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
    #plt.xlabel('Episodes')
    #plt.ylabel('Rewards')
    ax.plot(list_time_steps, values.T)
    
    min_std_values = np.add(values, -values_std)
    max_std_values = np.add(values, values_std)
    for i in range(num_agents):
        ax.fill_between(list_time_steps, min_std_values[i], max_std_values[i], alpha=0.5)
    #axs.set_title(f'update_period: {uP}, algorithm: {algo}')
    ax.set_title(f'Experiment name: {exp_name}')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Rewards')
    ax.set_ylim(0, 7)
    ax.legend(list_agents)

    plot_name = exp_name + '.png'
    path_plot = os.path.join(path_save, plot_name)
    figure.savefig(path_plot)
    return

def plot_total_reward(avg_reward: np.array,
    path_save: str,
    exp_name: str,
    ) -> None:
    arr_rewards_T = avg_reward
    values = arr_rewards_T
    
    num_time_steps = values.shape[1]
    list_time_steps = [i for i in range(num_time_steps)]
    num_algos = 6
    list_algos = ["IDDPG", "SNDDPG", "FLDDPG", "local_update", "round_update", "pair_update"]
    #list_agents = [f"Robot {i}" for i in range(num_agents)]
    
    figure, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
    #plt.xlabel('Episodes')
    #plt.ylabel('Rewards')
    ax.plot(list_time_steps, values.T)
    ax.set_title(f'Experiment name: {exp_name}')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Rewards')
    ax.set_ylim(-5, 5)
    ax.legend(list_algos)

    plot_name = exp_name + '.png'
    path_plot = os.path.join(path_save, plot_name)
    figure.savefig(path_plot)
    return
def generate_subdirs(path_parent, dict_key_value
    ) -> None:
    if os.path.isdir(path_parent):
        print("Parent directory exists.")
    else: raise ValueError("No parent directory exists.")

    algorithms = dict_key_value['algorithm']
    update_periods = dict_key_value['update_period']
    random_seeds = dict_key_value['random_seed']

    for algorithm in algorithms:
        path_algorithm = os.path.join(path_parent, algorithm)
        if not os.path.isdir(path_algorithm):
            os.makedirs(path_algorithm)
        for update_period in update_periods:
            path_update_period = os.path.join(path_algorithm, 'update_period_' + str(update_period))
            if not os.path.isdir(path_update_period):
                os.makedirs(path_update_period)
                for random_seed in random_seeds:
                    path_random_seed = os.path.join(path_update_period, str(random_seed))
                    if not os.path.isdir(path_random_seed):
                        os.makedirs(path_random_seed)
    return
    
def save_individual_reward_plot(path_parent):
    list_exp = os.listdir(path_parent)

    # Using a directory to read the file names.
    for exp_name in list_exp:
        path_log = get_path_log(path_parent, exp_name)
        path_reward = get_path_reward(path_parent, exp_name)
        is_path_exist = os.path.exists(path_reward)
        if is_path_exist == True:
            print(f"data is found at: {path_reward}")
            data_reward = np.load(path_reward)
            plot_reward(data_reward, path_log, exp_name)
            print(f"Plot is successfully saved in {path_log}")
        else: print(f"No data is found at: {path_reward}")

def save_averaged_reward_plot(path_parent, dict_exp):
    
    for algorithm in dict_exp.keys():
        for update_period, list_seed_name in dict_exp[algorithm].items():
            list_reward = []
            for seed_name, exp_name in dict_exp[algorithm][update_period].items():
                path_log = get_path_log(path_parent, exp_name)
                path_reward = get_path_reward(path_parent, exp_name)
                data_reward = np.load(path_reward)
                list_reward.append(data_reward)
            average_reward = np.average(list_reward, axis=0)
            std_reward = np.std(list_reward, axis=0)
            path_algo = os.path.join(path_parent, algorithm)
            path_algo_up = os.path.join(path_algo, update_period)
            plot_name = algorithm + '_' + update_period
            plot_reward_algo_up(average_reward, std_reward, path_algo_up, plot_name)

            print(f"Plot is successfully saved in every update_period subdir")
            #plot_reward(data_reward, path_log, exp_name)
    return   

def save_total_averaged_reward_plot(path_parent, dict_exp, dict_algorithm_up):
    
    list_reward_algos = []
    for algorithm in dict_exp.keys():
        list_exp_name = dict_exp[algorithm][dict_algorithm_up[algorithm]].values()
        list_reward = []
        for exp_name in list_exp_name:
            path_log = get_path_log(path_parent, exp_name)
            path_reward = get_path_reward(path_parent, exp_name)
            print(f"list_exp_name: {list_exp_name}")
            data_reward = np.load(path_reward)
            list_reward.append(data_reward)
        average_reward = np.average(list_reward, axis=0)
        print(average_reward.shape)
        average_reward_over_agents = np.average(average_reward, axis=1)
        average_reward_over_agents.reshape(80,1)
        print(average_reward_over_agents.shape)
        list_reward_algos.append(average_reward_over_agents)
    plot_name = 'total_reward_plot'
    plot_total_reward(np.array(list_reward_algos), path_parent, plot_name)
    print(f"Plot is successfully saved.")
            #plot_reward(data_reward, path_log, exp_name)
    return

def calculate_num_catastrophic_interference(array_reward, threshold_ci):
    '''
    input array of rewards (4 x 80)
    output - array of number of catastrophic interference (ci) for each agent (4 x 1)
    '''
    list_range_max = [np.maximum(agent_reward) for agent_reward in array_reward]
    list_range_min = [np.minimum(agent_reward) for agent_reward in array_reward]
    
    list_total_abs_change = []
    list_num_ci = []
    for agent_id, agent_reward in enumerate(array_reward):
        list_abs_change = []
        for episode in range(len(agent_reward)-1):
            abs_change = abs(agent_reward[episode] - agent_reward[episode+1])/(list_range_max[agent_id]-list_range_min[agent_id])
            list_abs_change.append(abs_change)
        array_abs_change = np.array(list_abs_change)
        list_total_abs_change.append(list_abs_change)

        num_ci = len(array_abs_change[array_abs_change > threshold_ci])
        list_num_ci.append(num_ci)
    return list_num_ci

def match_exp_setting(path_parent: str
    ) -> dict:
    '''
    input: path of parent directory
    output: dict mapping between dirs in path_parent with the parameter settings
    '''
    # Discard non-related dirs and files
    list_exp = os.listdir(path_parent)
    list_exp = [name_exp for name_exp in list_exp if len(name_exp) > 13]
	
    dict_exp = {}
    list_update_periods = ['update_period_1', 'update_period_3', 'update_period_5']
    list_random_seeds = ['101', '102', '103']
    # list_algo_name = []
    # for name_exp in list_exp:
    #     list_split = name_exp.split('-')
    #     algo_name = list_split[0]
    #     list_algo_name.append(algo_name)
    # list_algo_name.remove('SwarmDDPG')
    # list_algo_name.append('local_update')
    list_algo_name = ['IDDPG', 'SNDDPG', 'FLDDPG', 'local_update', 'round_update', 'pair_update'] 
	
    # Separate list_exp with algo_names
    list_IDDPG = [name_exp for name_exp in list_exp if 'IDDPG' in name_exp]
    list_SNDDPG = [name_exp for name_exp in list_exp if 'SNDDPG' in name_exp]
    list_FLDDPG = [name_exp for name_exp in list_exp if 'FLDDPG' in name_exp]
    list_SwarmDDPG = [name_exp for name_exp in list_exp if 'SwarmDDPG' in name_exp]
    list_local_update = []
    list_round_update = []
    list_pair_update = []
    count = 0
    for name_exp in list_SwarmDDPG:
        if count == 0:
            list_local_update.append(name_exp)
            count += 1
        elif count == 1:
            list_round_update.append(name_exp)
            count += 1
        elif count == 2:
            list_pair_update.append(name_exp)
            count = 0
	
    dict_files = {'IDDPG': list_IDDPG, 'SNDDPG': list_SNDDPG, 'FLDDPG': list_FLDDPG,
                 'local_update': list_local_update, 'round_update': list_round_update,
                 'pair_update': list_pair_update}
        
    
    # Initialise dict_exp
    dict_exp = dict.fromkeys(list_algo_name)

    # dict preparation for param, name_exp matching
    for algo_name in dict_exp.keys():
        dict_exp[algo_name] = dict.fromkeys(list_update_periods)
        for i, update_period in enumerate(list_update_periods):
            dict_exp[algo_name][update_period] = dict.fromkeys(list_random_seeds)
            for j, random_seeds in enumerate(list_random_seeds):
                index = len(list_random_seeds)*i+j
            
                #print(f"index: {index}")
                #print(f"dict_files[algo_name]: {dict_files[algo_name]}")
                dict_exp[algo_name][update_period][random_seeds] = dict_files[algo_name][index]
        
    return dict_exp


dict_IDDPG = {'update_period_1': ['IDDPG-20230222-200936',
                      'IDDPG-20230223-004434',
                      'IDDPG-20230223-052433'],
              'update_period_3': ['IDDPG-20230223-100613',
                      'IDDPG-20230223-143859',
                      'IDDPG-20230223-191829'],
              'update_period_5': ['IDDPG-20230223-234842',
                      'IDDPG-20230224-045714',
                      'IDDPG-20230224-102218']}
dict_SNDDPG = {'update_period_1': ['SNDDPG-20230222-214715',
                       'SNDDPG-20230223-022142',
                       'SNDDPG-20230223-070429'],
              'update_period_3': ['SNDDPG-20230223-113915',
                      'SNDDPG-20230223-161720',
                      'SNDDPG-20230223-205630'],
              'update_period_5': ['SNDDPG-20230224-012708',
                      'SNDDPG-20230224-065046',
                      'SNDDPG-20230224-121620']}
dict_FLDDPG = {'update_period_1': ['FLDDPG-20230222-230953',
                       'FLDDPG-20230223-034437',
                       'FLDDPG-20230223-082832'],
              'update_period_3': ['FLDDPG-20230223-130117',
                      'FLDDPG-20230223-174100',
                      'FLDDPG-20230223-221115'],
              'update_period_5': ['FLDDPG-20230224-030532',
                      'FLDDPG-20230224-082944']}
dict_local_update = {'update_period_1': ['SwarmDDPG-20230222-201113',
                             'SwarmDDPG-20230222-232657',
                             'SwarmDDPG-20230223-023904'],
              'update_period_3': ['SwarmDDPG-20230223-055714',
                      'SwarmDDPG-20230223-091550',
                      'SwarmDDPG-20230223-122657'],
              'update_period_5': ['SwarmDDPG-20230223-154414',
                      'SwarmDDPG-20230223-185850',
                      'SwarmDDPG-20230223-221257']}
dict_round_update = {'update_period_1': ['SwarmDDPG-20230222-214845',
                             'SwarmDDPG-20230223-010202',
                             'SwarmDDPG-20230223-041703'],
              'update_period_3': ['SwarmDDPG-20230223-073531',
                     'SwarmDDPG-20230223-105045',
                     'SwarmDDPG-20230223-140444'],
              'update_period_5': ['SwarmDDPG-20230223-172029',
                      'SwarmDDPG-20230223-203537',
                      'SwarmDDPG-20230223-221257']}
dict_pair_update = {'update_period_1': ['SwarmDDPG-20230224-161632',
                            'SwarmDDPG-20230224-180631',
                            'SwarmDDPG-20230224-195743'],
              'update_period_3': ['SwarmDDPG-20230224-214856',
                      'SwarmDDPG-20230224-233835', 
                      'SwarmDDPG-20230225-012700'],
              'update_period_5': ['SwarmDDPG-20230225-030542',
                      'SwarmDDPG-20230225-044001',
                      'SwarmDDPG-20230225-061518']}

dict_exp = {'IDDPG': dict_IDDPG, 'SNDDPG': dict_SNDDPG, 'FLDDPG': dict_FLDDPG,
            'local_update': dict_local_update, 'round_update': dict_round_update,
            'pair_update': dict_pair_update}
# This should be manually written, as I choose it manually
dict_algorithm_up = {'IDDPG': 'update_period_1', 'SNDDPG': 'update_period_3',
                     'FLDDPG': 'update_period_1', 'local_update': 'update_period_1',
                     'round_update': 'update_period_1', 'pair_update': 'update_period_1'}
if __name__=="__main__":
    # PARSE and their descriptions
    parser = argparse.ArgumentParser(
        description='Experiment script for fl4sr project.')
    parser.add_argument(
        '--name_parent_dir',
        type=str,
        help='a name of parent directory containing all the experiment subdirs')
    args = parser.parse_args()

    HOME = os.environ['HOME']
    path_data = HOME + '/catkin_ws/src/fl4sr/src/data'
    dir_name = args.name_parent_dir
    path_parent = os.path.join(path_data, dir_name)
    dict_key_value = {'algorithm': ['IDDPG', 'SNDDPG', 'FLDDPG', 'local_update', 'round_update', 'pair_update'], 
                      'update_period': [1,3,5],
                      'random_seed': [101, 102, 103]}

    
    save_individual_reward_plot(path_parent)
    # generate_subdirs(path_parent, dict_key_value)
    # dict_exp = match_exp_setting(path_parent)
    # print(f"dict_exp: {dict_exp}")
    # save_averaged_reward_plot(path_parent, dict_exp)
    # save_total_averaged_reward_plot(path_parent, dict_exp, dict_algorithm_up)
    
    


    # Using csv files to read the file_name
    #csv_files = open("plot_reward_exps.csv", 'r')
    #list_exp = list(csv.reader(csv_files, delimiter=","))
    #csv_files.close()
    print(f"Plots are successfully generated!")

    

