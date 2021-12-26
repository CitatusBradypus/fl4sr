#! /usr/bin/env python3.8
import sys
import os
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')
import numpy as np
import time
import pickle
from Enviroment import Enviroment
from worlds import World
from DDPG import DDPG
from buffers import BasicBuffer, Transition

class IndividualDDPG():
    """Individial DDPG algorithm. Serves as basis for other algorithms.
    """

    def __init__(self,
        episode_count: int,
        episode_step_count: int,
        world: World
        ) -> None:
        # global like variables
        self.TIME_TRAIN = 5
        self.TIME_TARGET = 5
        self.TIME_UPDATE = 10
        self.TIME_LOGGER = 16
        self.TIME_SAVE = 50
        # random actions
        self.EPSILON = 0.9
        self.EPSILON_DECAY = 0.99995
        # init experiment and error values
        self.episode_count = episode_count
        self.episode_step_count = episode_step_count
        self.episode_error = 0
        self.episode_step_error = 0
        # init some world values
        self.robot_count = world.robot_count
        # TODO seed random / numpy random
        # init enviroment and dimensions
        self.enviroment = Enviroment(world)
        self.observation_dimension = self.enviroment.observation_dimension
        self.action_dimension = self.enviroment.action_dimension
        # init buffers and agents
        self.BUFFER_SIZE = 50000
        self.buffers = self.init_buffers()
        self.agents = self.init_agents()
        # loggers
        self.NAME = 'IndividualDDPG'
        self.init_data()
        # debugging
        self.debug = False
        return

    def init_buffers(self
        ) -> list:
        """Creates list with buffers.

        Returns:
            list: Buffers list.
        """
        return [BasicBuffer(self.BUFFER_SIZE) 
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
        self.average_rewards = np.zeros((self.episode_count, self.robot_count))
        self.robots_succeeded_once = np.zeros((self.episode_count, self.robot_count), dtype=bool)        
        return

    def run(self
        ) -> tuple:
        self.init_paths()
        self.parameters_save()
        self.print_starting_info()
        if self.episode_step_error != 0:
            # print('SUBSCRIBERS')
            # self.enviroment.init_subscribers()
            pass
        for episode in range(self.episode_error, self.episode_count):
            self.enviroment.reset()
            current_states = self.enviroment.get_starting_states()
            total_rewards = np.zeros(self.robot_count)
            for step in range(self.episode_step_error, self.episode_step_count):
                actions = self.agents_actions(current_states)
                actions = self.actions_add_random(actions)
                new_states, rewards, robots_finished, robots_succeeded_once, error = self.enviroment.step(actions)
                if error:
                    self.episode_error = episode
                    self.episode_step_error = step
                    print('ERROR: DDPG: Death robot detected during {}.{}'.format(episode, step))
                    return False, episode, step
                total_rewards += rewards
                self.buffers_save_transitions(current_states, actions, rewards, new_states, robots_finished)
                if step % self.TIME_TRAIN == 0:
                    self.agents_train()
                if step % self.TIME_TARGET == 0:
                    self.agents_target()
                if step % self.TIME_LOGGER == 0:
                    print('{}.{}'.format(episode, step))
                current_states = new_states
            self.data_collect(episode, total_rewards, robots_succeeded_once)
            print('Average episode rewards: {}'.format(self.average_rewards[episode]))
            if episode % self.TIME_UPDATE == 0:
                self.agents_update()
            if episode % self.TIME_SAVE == 0:
                self.agents_save(episode)
                self.data_save(episode)
        self.enviroment.reset()
        self.agents_save()
        self.data_save()
        return True, None, None

    def test(self
        ) -> None:
        self.print_starting_info(training=False)
        for episode in range(self.episode_count):
            self.enviroment.reset()
            current_states = self.enviroment.get_starting_states()
            total_rewards = np.zeros(self.robot_count)
            for step in range(self.episode_step_count):
                actions = self.agents_actions(current_states)
                new_states, rewards, robots_finished, robots_succeeded_once = self.enviroment.step(actions)
                total_rewards += rewards
                if step % self.TIME_LOGGER == 0:
                    print('{}.{}'.format(episode, step))
                current_states = new_states
            self.data_collect(episode, total_rewards, robots_succeeded_once)
            print('Average episode rewards: {}'.format(self.average_rewards[episode]))
            if episode % self.TIME_SAVE == 0:
                self.data_save(episode)
        self.data_save()
        return

    def agents_actions(self,
        states: np.ndarray,
        ) -> np.ndarray:
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
        for i in range(self.robot_count):
            self.buffers[i].add([Transition(s[i], a[i], r[i], s_[i], f[i])])
        return

    def actions_add_random(self, 
        actions: np.ndarray
        ) -> np.ndarray:
        # get current actions
        angles_a = actions[:, 0]
        linears_a = actions[:, 1]
        # where to use randoms and generate them
        randoms = np.random.uniform(0, 1, self.robot_count)
        use_randoms = np.where(randoms < self.EPSILON, 1, 0)
        angles_r = use_randoms * np.random.uniform(-0.5, 0.5, self.robot_count)
        linears_r = use_randoms * np.random.uniform(0, 0.5, self.robot_count)
        # add randoms and clip
        angles = angles_a + angles_r
        linears = linears_a + linears_r
        angles = np.clip(angles, -1, 1)
        linears = np.clip(linears, 0, 1)
        new_actions = np.array((angles, linears)).T
        # update epsilon
        self.EPSILON *= self.EPSILON_DECAY
        return new_actions

    def agents_train(self):
        for agent in self.agents:
            agent.train() 
        return

    def agents_target(self):
        for agent in self.agents:
            agent.update_targets() 
        return

    def agents_update(self):
        return

    def agents_save(self, 
        episode:int=None
        ) -> None:
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
        self.average_rewards[episode] = total_rewards / self.episode_step_count
        self.robots_succeeded_once[episode] = robots_succeeded_once
        if episode != 0:
            same_indexes = np.where(self.average_rewards[episode-1] == self.average_rewards[episode])[0]
            if len(same_indexes) > 0:
                self.debug = True
                print('ERROR: Suspicious behaviour discovered, repeated fitness for robots {}'.format(same_indexes))
        return

    def data_save(self, 
        episode:int=None
        ) -> None:
        if episode is None:
            episode = 'final'
        np.save(self.path_log + '/rewards'.format(), 
                self.average_rewards)
        np.save(self.path_log + '/succeded'.format(),
                self.robots_succeeded_once)
        return

    def parameters_save(self
        ) -> None:
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
        parameters['LEARNING_RATE'] = self.agents[0].LEARNING_RATE
        parameters['BATCH_SIZE'] = self.agents[0].BATCH_SIZE
        parameters['GAMMA'] = self.agents[0].GAMMA
        parameters['TAU_TARGET'] = self.agents[0].TAU
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
        print('{}'.format(self.NAME))
        print('----------------')
        print('Episodes = {}'.format(self.episode_count))
        print('Steps per episode = {}'.format(self.episode_step_count))
        print('Training = {}'.format(training))
        print('Buffers = {}'.format(len(self.buffers)))
        print('Agents = {}'.format(len(self.agents)))
        print('----------------')
        return
