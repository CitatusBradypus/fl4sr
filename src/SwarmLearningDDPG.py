#! /usr/bin/env python3.8
from collections import namedtuple
import sys
import os
from numpy.core.fromnumeric import size

import torch
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')
from IndividualDDPG_limit import IndividualDDPG
from worlds import World
from DDPG_limit import DDPG
from buffers import BasicBuffer, Transition
import numpy as np


Means = namedtuple('Averages', 'aw, ab, cw, cb')


class SwarmLearningDDPG(IndividualDDPG):
    """Swarm learning DDPG with prepared soft update.

    Args:
        IndividualDDPG (): ...
    """

    def __init__(self, 
        episode_count: int,
        episode_step_count: int,
        world: World,
        env = 'Enviroment',
        reward_goal: float = 100.0,
        reward_collision: float = -10.0,
        reward_progress: float = 40.0,
        reward_max_collision: float = 3.0,
        list_reward: int = 1,
        factor_linear: float = 0.25,
        factor_angular: float = 1.0,
        discount_factor: float = 0.99,
        is_progress: bool = False,
        name=None, 
        ) -> None:
        """Initialize

        Args:
            episode_count (int): ...
            episode_step_count (int): ...
            world (World): contains information about experiment characteristics
        """
        # set hyperparameters
        self.NAME = 'SwarmDDPG'
        self.BUFFER_SIZE = 10000
        super().__init__(episode_count, episode_step_count, world)
        # get model counts
        self.agents_count = len(self.agents)
        self.actor_layers_count = len(self.agents[0].actor.layers)
        self.critic_layers_count = len(self.agents[0].critic.layers)
        # averaging params
        self.TAU = 0.5

        # simFL counter [0-4]
        self.update_counter = 0
        return

    def check_counter(self, 
        update_counter: int
        ) -> int:

        if update_counter >= 4:
            update_counter = 0
        return update_counter

    def agents_update(self,
        reward: np.ndarray
        ) -> None:
        """Perform update of agents.

        Args:
            reward (np.ndarray): average rewards obtained by agents between updates
        """
        self.local_swarm_update()
        return

    def local_swarm_update(self):
        list_means = [ [] for _ in range(self.agents_count)]

        for i in range(self.agents_count):
            if i == 0:
                list_ids = [0, 1, self.agents_count-1]
            elif i == self.agents_count-1:
                list_ids = [0, self.agents_count-1, self.agents_count-2]
            else: 
                list_ids = [i-1, i, i+1]
            means = self.get_means(list_ids)
            list_means[i] = means

        self.agents_update_models(list_means)

    def simFL_update(self):
        
        list_means = [ [] for _ in range(self.agents_count)]

        for i in range(self.agents_count):
            list_ids = [self.update_counter, i]
            means = self.get_means(list_ids)
            list_means[i] = means

        self.update_counter += 1
        self.update_counter = self.check_counter(self.update_counter)

        self.agents_update_models(list_means)

        
    def get_means(self,
        list_ids: list
        ) -> tuple:
        """Compute mean parameter values of given agents.

        Returns:
            tuple: Averages
        """
        # compute the number of agents
        num_agents = len(list_ids)

        # init values
        actor_mean_weights = [None] * self.actor_layers_count
        actor_mean_bias = [None] * self.actor_layers_count
        critic_mean_weights = [None] * self.critic_layers_count
        critic_mean_bias = [None] * self.critic_layers_count
        for i in range(self.actor_layers_count):
            actor_mean_weights[i] = torch.zeros(size=self.agents[0].actor.layers[i].weight.shape).cuda()
            actor_mean_bias[i] = torch.zeros(size=self.agents[0].actor.layers[i].bias.shape).cuda()
        for i in range(self.critic_layers_count):
            critic_mean_weights[i] = torch.zeros(size=self.agents[0].critic.layers[i].weight.shape).cuda()
            critic_mean_bias[i] = torch.zeros(size=self.agents[0].critic.layers[i].bias.shape).cuda()
        # compute means
        with torch.no_grad():
            for i in range(self.actor_layers_count):
                for j in list_ids:
                    actor_mean_weights[i] += self.agents[j].actor.layers[i].weight.data.clone()
                    actor_mean_bias[i] += self.agents[j].actor.layers[i].bias.data.clone()
                actor_mean_weights[i] = actor_mean_weights[i] / num_agents
                actor_mean_bias[i] = actor_mean_bias[i] / num_agents
            for i in range(self.critic_layers_count):
                for j in list_ids:
                    critic_mean_weights[i] += self.agents[j].critic.layers[i].weight.data.clone()
                    critic_mean_bias[i] += self.agents[j].critic.layers[i].bias.data.clone()
                critic_mean_weights[i] = critic_mean_weights[i] / num_agents
                critic_mean_bias[i] = critic_mean_bias[i] / num_agents
        return Means(actor_mean_weights, actor_mean_bias, 
                     critic_mean_weights, critic_mean_bias)

    def agents_update_models(self, 
        list_means: list
        ) -> None:
        """Update parameter values of existing agents using computed means and soft udpate.

        Args:
            list_means (list): List of tuples holding locally averaged parameters.
        """
        for i in range(self.agents_count):
            for j in range(self.actor_layers_count):
                self.agents[i].actor.layers[j].weight.data = \
                    (1 - self.TAU) * self.agents[i].actor.layers[j].weight.data + self.TAU * list_means[i].aw[j].clone()
                self.agents[i].actor.layers[j].bias.data = \
                    (1 - self.TAU) * self.agents[i].actor.layers[j].bias.data + self.TAU * list_means[i].ab[j].clone()
            for j in range(self.critic_layers_count):
                self.agents[i].critic.layers[j].weight.data = \
                    (1 - self.TAU) * self.agents[i].critic.layers[j].weight.data + self.TAU * list_means[i].cw[j].clone()
                self.agents[i].critic.layers[j].bias.data = \
                    (1 - self.TAU) * self.agents[i].critic.layers[j].bias.data + self.TAU * list_means[i].cb[j].clone()
        return

    