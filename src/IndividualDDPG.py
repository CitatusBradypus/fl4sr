#! /usr/bin/env python3.8
import sys
import os
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')
import numpy as np
import time
import pickle
from Enviroment import Enviroment
from environment_real import RealEnviroment
from worlds import World
from DDPG import DDPG
from buffers import BasicBuffer, PrioritizedExperienceReplayBuffer, Transition


class IndividualDDPG():
    """Individial DDPG algorithm. Serves as basis for other algorithms.
    """

    def __init__(self,
        episode_count: int,
        episode_step_count: int,
        world: World,
        env = 'Enviroment',
        name=None
        ) -> None:
        """Initialize class and whole experiment.

        Args:
            episode_count (int): ...
            episode_step_count (int): ...
            world (World): contains information about experiment characteristics
            name (str, optional): Name of used method. Defaults to None.
        """
        # global like variables
        self.TIME_TRAIN = 5
        self.TIME_TARGET = 5
        self.EPISODE_UPDATE = True
        self.TIME_UPDATE = 2
        self.TIME_LOGGER = 16
        self.TIME_SAVE = 25
        # random actions
        self.EPSILON = 0.9
        self.EPSILON_DECAY = 0.99997
        # init experiment and error values
        self.episode_count = episode_count
        self.episode_step_count = episode_step_count
        self.episode_error = 0
        self.episode_step_error = 0
        # init some world values
        self.robot_count = world.robot_count
        # init enviroment and dimensions
        self.world = world
        self.env = env
        self.init_enviroment()
        # init buffers and agents
        self.BUFFER_TYPE = BasicBuffer
        if not hasattr(self, 'BUFFER_SIZE'):
            self.BUFFER_SIZE = 30000
        self.buffers = self.init_buffers()
        self.agents = self.init_agents()
        # loggers
        if not hasattr(self, 'NAME'):
            if name is not None:
                self.NAME = name
            else:
                self.NAME = 'IDDPG'
        self.init_data()
        # debugging
        self.debug = False
        print(self.buffers)
        print(self.agents)
        # paths
        self.init_paths()
        return

    def init_enviroment(self
        ) -> None:
        """Initializes environment.
        """
        if self.env == 'Enviroment'
            self.enviroment = Enviroment(self.world)
        elif self.env == 'RealEnviroment'
            self.enviroment = RealEnviroment(self.world)
        else: raise Exception(f"No Environment named {self.env} is available.")
        self.observation_dimension = self.enviroment.observation_dimension
        self.action_dimension = self.enviroment.action_dimension
        return

    def init_buffers(self
        ) -> list:
        """Creates list with buffers.

        Returns:
            list: Buffers list.
        """
        return [self.BUFFER_TYPE(self.BUFFER_SIZE) 
                for i in range(self.robot_count)]

    def init_agents(self
        ) -> list:
        """Creates list with agents.

        Returns:
            list: Agents list.
        """
        return [DDPG(self.buffers[i], 
                     self.observation_dimension, 
                     self.action_dimension) 
                for i in range(self.robot_count)]

    def init_paths(self):
        """Initializes and creates file system for saving obtained information.
        """
        path_data = HOME + '/catkin_ws/src/fl4sr/src/data'
        name_run = self.NAME + '-' + time.strftime("%Y%m%d-%H%M%S")
        self.path_run = path_data + '/' + name_run
        self.path_weights = self.path_run + '/weights'
        self.path_log = self.path_run + '/log'
        if not os.path.exists(self.path_weights):
            os.makedirs(self.path_weights, exist_ok=True)
        if not os.path.exists(self.path_log):
            os.makedirs(self.path_log, exist_ok=True)
        return

    def init_data(self
        ) -> None:
        """Initializes data containers.
        """
        self.average_rewards = np.zeros((self.episode_count, self.robot_count))
        self.robots_succeeded_once = np.zeros((self.episode_count, self.robot_count), dtype=bool)        
        self.robots_finished = np.zeros((self.episode_count, self.robot_count), dtype=bool)
        self.data = []      
        return

    def init_data_test(self
        ) -> None:
        """Initializes data containers for evaluation.
        """
        self.robots_succeeded_once = np.zeros((self.episode_step_count, self.robot_count), dtype=bool)        
        self.robots_finished = np.zeros((self.episode_step_count, self.robot_count), dtype=bool)
        self.data = []      
        return
    def init_data_real(self
        ) -> None:
        """Initializes data containers for evaluation.
        """
        self.robots_succeeded_once = np.zeros((self.episode_step_count, self.robot_count), dtype=bool)        
        self.robots_finished = np.zeros((self.episode_step_count, self.robot_count), dtype=bool)
        self.data = []
        self.exp_time = {}
        return

    def terminate_enviroment(self):
        """Sets enviroment to None. 
        Used before saving whole class when error is encountered.
        """
        self.enviroment = None
        return

    def run(self
        ) -> tuple:
        """Runs learning experiment.

        Returns:
            tuple: bool success (no errors encoutered), error episode, error step
        """
        # before start
        self.parameters_save()
        self.print_starting_info()
        total_rewards = np.zeros(self.robot_count)
        # epizode loop
        for episode in range(self.episode_error, self.episode_count):
            self.enviroment.reset()
            current_states = self.enviroment.get_current_states()
            data_total_rewards = np.zeros(self.robot_count)
            if self.episode_error != episode:
                self.episode_step_error = 0
            for step in range(self.episode_step_error, self.episode_step_count):
                # get actions
                actions = self.agents_actions(current_states)
                actions = self.actions_add_random(actions, episode)
                # perform step
                new_states, rewards, robots_finished, robots_succeeded_once, error, _ = self.enviroment.step(actions)
                if error:
                    self.episode_error = episode
                    self.episode_step_error = step
                    print('ERROR: DDPG: Death robot detected during {}.{}'.format(episode, step))
                    return False, episode, step
                total_rewards += rewards
                data_total_rewards += rewards
                self.buffers_save_transitions(current_states, actions, rewards, new_states, robots_finished)
                # train
                if step % self.TIME_TRAIN == 0:
                    self.agents_train()
                # update target
                if step % self.TIME_TARGET == 0:
                    self.agents_target()
                # step federated update
                if (not self.EPISODE_UPDATE) and step % self.TIME_UPDATE == 0:
                    print('UPDATE')
                    mean_rewards = total_rewards / self.TIME_UPDATE
                    self.agents_update(mean_rewards)
                    total_rewards = np.zeros(self.robot_count)
                # print info
                if step % self.TIME_LOGGER == 0:
                    print('{}.{}'.format(episode, step))
                    print(actions)
                current_states = new_states
            self.data_collect(episode, data_total_rewards, robots_succeeded_once)
            # print info
            print('Average episode rewards: {}'.format(self.average_rewards[episode]))
            # episode federated update
            if self.EPISODE_UPDATE and episode % self.TIME_UPDATE == 0:
                print('UPDATE')
                mean_rewards = total_rewards / (self.TIME_UPDATE * self.episode_step_count)
                self.agents_update(mean_rewards)
                total_rewards = np.zeros(self.robot_count)
            # save data
            if episode % self.TIME_SAVE == 0:
                self.agents_save(episode)
                self.data_save(episode)
        self.enviroment.reset()
        self.agents_save()
        self.data_save()
        return True, None, None

    def test(self
        ) -> tuple:
        """Runs evaluation experiment.

        Returns:
            tuple: bool success (no errors encountered), error episode, error step
        """
        # before start
        self.init_data_test()
        self.parameters_save()
        self.print_starting_info(False)
        # epizode loop
        for episode in range(self.episode_error, self.episode_count):
            self.enviroment.reset()
            self.init_data_test()
            current_states = self.enviroment.get_current_states()
            if self.episode_error != episode:
                self.episode_step_error = 0
            for step in range(0, self.episode_step_count):
                actions = self.agents_actions(current_states)
                new_states, rewards, robots_finished, robots_succeeded_once, error, data = self.enviroment.step(actions)
                if error:
                    self.episode_error = episode
                    self.episode_step_error = step
                    print('ERROR: DDPG: Death robot detected during {}.{}'.format(episode, step))
                    return False, episode, step
                if step % self.TIME_LOGGER == 0:
                    print('{}.{}'.format(episode, step))
                    print(actions)
                current_states = new_states
                self.data_collect_test(step, robots_finished, robots_succeeded_once, data)
                if np.any(robots_finished):
                    break
            print('Robots succeded once: {}'.format(robots_succeeded_once))
            self.data_save_test(episode)
        self.enviroment.reset()
        return True, None, None

    def test_real(self
        ) -> tuple:
        """Runs evaluation experiment.

        Returns:
            tuple: bool success (no errors encountered), error episode, error step
        """
        # before start
        self.init_data_real()
        self.parameters_save()
        self.print_starting_info(False)
        # epizode loop
        for episode in range(self.episode_error, self.episode_count):
            self.enviroment.reset()
            self.init_data_real()
            current_states = self.enviroment.get_current_states()
            if self.episode_error != episode:
                self.episode_step_error = 0
            start_time = time.time()
            for step in range(0, self.episode_step_count):
                actions = self.agents_actions(current_states)
                new_states, rewards, robots_finished, robots_succeeded_once, error, data = self.enviroment.step(actions)
                if error:
                    self.episode_error = episode
                    self.episode_step_error = step
                    print('ERROR: DDPG: Death robot detected during {}.{}'.format(episode, step))
                    return False, episode, step
                if step % self.TIME_LOGGER == 0:
                    print('{}.{}'.format(episode, step))
                    print(actions)
                current_states = new_states
                self.data_collect_test(step, robots_finished, robots_succeeded_once, data)
                if np.any(robots_finished):
                    break
            self.exp_time[f'{episode}'] = time.time()-start_time
            print('Robots succeded once: {}'.format(robots_succeeded_once))
            self.data_save_real(episode)
        return True, None, None

    

    def agents_actions(self,
        states: np.ndarray,
        ) -> np.ndarray:
        """Get actions of all agents.

        Args:
            states (np.ndarray): ...

        Returns:
            np.ndarray: actions
        """
        actions = []
        for i in range(self.robot_count):
            actions.append(self.agents[i].select_action(states[i]))
        actions = np.array(actions)
        return actions
        
    def buffers_save_transitions(self, 
        s: np.ndarray, 
        a: np.ndarray, 
        r: np.ndarray, 
        s_: np.ndarray, 
        f: np.ndarray
        ) -> None:
        """Save transitions to buffers.
        Args as described in thesis, only "D" is named as "f".

        Args:
            s (np.ndarray): ...
            a (np.ndarray): ...
            r (np.ndarray): ...
            s_ (np.ndarray): ...
            f (np.ndarray): ...
        """
        for i in range(self.robot_count):
            self.buffers[i].add([Transition(s[i], a[i], r[i], s_[i], f[i])])
        return

    def actions_add_random(self, 
        actions: np.ndarray,
        episode: int
        ) -> np.ndarray:
        """Add random actions.

        Args:
            actions (np.ndarray): ...
            episode (int): not used

        Returns:
            np.ndarray: actions possibly with some actions randomized
        """
        # get current actions
        angles_a = actions[:, 0]
        linears_a = actions[:, 1]
        # where to use randoms and generate them
        randoms = np.random.uniform(0, 1, self.robot_count)
        use_randoms = np.where(randoms < self.EPSILON, 1, 0)
        angles_r = np.random.uniform(-1, 1, self.robot_count)
        linears_r = np.random.uniform(0, 1, self.robot_count)
        # add randoms and clip
        angles = (1 - use_randoms) * angles_a + use_randoms * angles_r
        linears = (1 - use_randoms) * linears_a + use_randoms * linears_r
        angles = np.clip(angles, -1, 1)
        linears = np.clip(linears, 0, 1)
        # combine new actions
        new_actions = np.array((angles, linears)).T
        # update random rate
        self.EPSILON *= self.EPSILON_DECAY
        return new_actions


    def agents_train(self):
        """Train all agents.
        """
        for agent in self.agents:
            agent.train() 
        return

    def agents_target(self):
        """Update target networks of all agents.
        """
        for agent in self.agents:
            agent.update_targets() 
        return

    def agents_update(self, rewards):
        """Update parameters of agents.
        (Obviously empty for IDDPG, SEDDPG, and SNDDPG)

        Args:
            rewards (np.array): average rewards obtained by agents between updates 
        """
        return

    def agents_save(self, 
        episode:int=None
        ) -> None:
        """Save weights of agents.

        Args:
            episode (int, optional): Current episode for file naming. Defaults to None.
        """
        if episode is None:
            episode = 'final'
        for i in range(len(self.agents)):
            self.agents[i].weights_save(self.path_weights + '/actor-{}-{}.pkl'
                                                            .format(episode, i),
                                        self.path_weights + '/critic-{}-{}.pkl'
                                                            .format(episode, i))
        return

    def agents_load(self, 
        paths_actor: list, 
        paths_critic: list
        ) -> None:
        """Load weights of agents.

        Args:
            paths_actor (list): list of paths to actors
            paths_critic (list): list of paths to critics
        """
        assert len(paths_actor) == len(paths_critic) == len(self.agents), 'Wrong load size!'
        for i in range(len(self.agents)):
            self.agents[i].weights_load(paths_actor[i], 
                                        paths_critic[i])
            self.agents[i].update_targets()
        return

    def data_collect(self, 
        episode,
        total_rewards, 
        robots_succeeded_once
        ) -> None:
        """Collect data from learning experiments.

        Args:
            episode (int): ...
            total_rewards (np.ndarray): ...
            robots_succeeded_once (np.ndarray): ...
        """
        self.average_rewards[episode] = total_rewards / self.episode_step_count
        self.robots_succeeded_once[episode] = robots_succeeded_once
        if episode != 0:
            same_indexes = np.where(self.average_rewards[episode-1] == self.average_rewards[episode])[0]
            if len(same_indexes) > 0:
                self.debug = True
                print('ERROR: Suspicious behaviour discovered, repeated fitness for robots {}'.format(same_indexes))
        return

    def data_collect_test(self, 
        step,
        robots_finished, 
        robots_succeeded_once,
        data
        ) -> None:
        """Collect data from evaluating experiments.

        Args:
            step (int): ...
            robots_finished (np.ndarray): ...
            robots_succeeded_once (np.ndarray): ...
            data (_type_): additional collected information for each step
        """
        self.robots_finished[step] = robots_finished
        self.robots_succeeded_once[step] = robots_succeeded_once
        self.data.append(data)
        return

    

    def data_save(self, 
        episode:int=None
        ) -> None:
        """Save collected data form learning.

        Args:
            episode (int, optional): ... . Defaults to None.
        """
        np.save(self.path_log + '/rewards'.format(), 
                self.average_rewards)
        np.save(self.path_log + '/succeded'.format(),
                self.robots_succeeded_once)
        return

    def data_save_test(self, 
        episode:int=None
        ) -> None:
        """Save collected data from evaluating.

        Args:
            episode (int, optional): ... . Defaults to None.
        """
        np.save(self.path_log + '/finished-{}'.format(episode), 
                self.robots_finished)
        np.save(self.path_log + '/succeded-{}'.format(episode),
                self.robots_succeeded_once)
        with open(self.path_log + '/data-{}.pkl'.format(episode), 'wb') as f:
            pickle.dump(self.data, f)
        return
        
        
    def data_save_real(self, 
        episode:int=None
        ) -> None:
        """Save collected data from evaluating.

        Args:
            episode (int, optional): ... . Defaults to None.
        """
        np.save(self.path_log + '/finished-{}'.format(episode), 
                self.robots_finished)
        np.save(self.path_log + '/succeded-{}'.format(episode),
                self.robots_succeeded_once)
        np.save(self.path_log + '/exp_time-{}'.format(episode),
                self.exp_time)
        with open(self.path_log + '/data-{}.pkl'.format(episode), 'wb') as f:
            pickle.dump(self.data, f)
        return
    def parameters_save(self
        ) -> None:
        """Save used parameters to file.
        """
        parameters = {}
        # method parameters
        parameters['NAME'] = self.NAME
        parameters['robot_count'] = self.robot_count
        parameters['observation_dimension'] = self.observation_dimension
        parameters['action_dimension'] = self.action_dimension
        parameters['episode_count'] = self.episode_count
        parameters['episode_step_count'] = self.episode_step_count        
        parameters['TIME_TRAIN'] = self.TIME_TRAIN
        parameters['TIME_TARGET'] = self.TIME_TARGET
        parameters['TIME_UPDATE'] = self.TIME_UPDATE
        parameters['EPSILON'] = self.EPSILON
        parameters['EPSILON_DECAY'] = self.EPSILON_DECAY
        parameters['buffer_count'] = len(self.buffers)
        parameters['BUFFER_SIZE'] = self.BUFFER_SIZE
        parameters['agent_count'] = len(self.agents)
        if self.NAME == 'FederatedLearningDDPG':
            parameters['TAU_UPDATE'] = self.TAU
        # ddpg parameters
        parameters['ACTOR_HIDDEN_LAYERS'] = self.agents[0].ACTOR_HIDDEN_LAYERS
        parameters['CRITIC_HIDDEN_LAYERS'] = self.agents[0].CRITIC_HIDDEN_LAYERS
        parameters['LEARNING_RATE_ACTOR'] = self.agents[0].LEARNING_RATE_ACTOR
        parameters['LEARNING_RATE_CRITIC'] = self.agents[0].LEARNING_RATE_CRITIC
        parameters['BATCH_SIZE'] = self.agents[0].BATCH_SIZE
        parameters['GAMMA'] = self.agents[0].GAMMA
        parameters['TAU_TARGET'] = self.agents[0].RHO
        # enviromental params
        parameters['COLLISION_RANGE'] = self.enviroment.COLLISION_RANGE
        parameters['GOAL_RANGE'] = self.enviroment.GOAL_RANGE
        parameters['PROGRESS_REWARD_FACTOR'] = self.enviroment.PROGRESS_REWARD_FACTOR
        parameters['REWARD_GOAL'] = self.enviroment.REWARD_GOAL
        parameters['REWARD_COLLISION'] = self.enviroment.REWARD_COLLISION
        # save parameters
        with open(self.path_log + '/parameters.pkl', 'wb+') as f:
            pickle.dump(parameters, f)
        return

    def print_starting_info(self, 
        training: bool=True
        ) -> None:
        """Print staring information about experiment.

        Args:
            training (bool, optional): ... . Defaults to True.
        """
        print('{}'.format(self.NAME))
        print('----------------')
        print('Episodes = {}'.format(self.episode_count))
        print('Steps per episode = {}'.format(self.episode_step_count))
        print('Running robots = {}'.format(self.world.robot_alives))
        print('Training = {}'.format(training))
        print('Buffers = {}'.format(len(self.buffers)))
        print('Buffer size = {}'.format(self.BUFFER_SIZE))
        print('Agents = {}'.format(len(self.agents)))
        print('----------------')
        return
