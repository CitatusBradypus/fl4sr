#! /usr/bin/env python3.8
from collections import namedtuple
import sys
import os
from numpy.core.fromnumeric import size

import torch
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')
from IndividualDDPG import IndividualDDPG
from worlds import World
from DDPG import DDPG
from buffers import BasicBuffer, Transition
import numpy as np


Means = namedtuple('Averages', 'aw, ab, cw, cb')


class FederatedLearningDDPG(IndividualDDPG):
    """Federated learning DDPG with prepared soft update.

    Args:
        IndividualDDPG (): ...
    """

    def __init__(self, 
        episode_count: int,
        episode_step_count: int,
        world: World,
        env = 'Enviroment',
        name=None, 
        reward_goal: float = 100.0,
        reward_collision: float = -10.0,
        reward_progress: float = -40.0,
        factor_linear: float = 0.25,
        factor_angular: float = 1.0
        
        ) -> None:
        """Initialize

        Args:
            episode_count (int): ...
            episode_step_count (int): ...
            world (World): contains information about experiment characteristics
        """
        # set hyperparameters
        self.NAME = 'FLDDPG'
        self.BUFFER_SIZE = 10000
        super().__init__(episode_count, episode_step_count, world, env, reward_goal, reward_collision, reward_progress, factor_linear, factor_angular)
        # get model coutns
        self.agents_count = len(self.agents)
        self.actor_layers_count = len(self.agents[0].actor.layers)
        self.critic_layers_count = len(self.agents[0].critic.layers)
        # averaging params
        self.TAU = 0.5
        return

    def agents_update(self,
        reward: np.ndarray
        ) -> None:
        """Perform update of agents.

        Args:
            reward (np.ndarray): average rewards obtained by agents between updates
        """
        means = self.get_means()
        self.agents_update_models(means)
        return

    def get_means(self
        ) -> tuple:
        """Compute mean parameter values of agents.

        Returns:
            tuple: Averages
        """
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
                for j in range(self.agents_count):
                    actor_mean_weights[i] += self.agents[j].actor.layers[i].weight.data.clone()
                    actor_mean_bias[i] += self.agents[j].actor.layers[i].bias.data.clone()
                actor_mean_weights[i] = actor_mean_weights[i] / self.agents_count
                actor_mean_bias[i] = actor_mean_bias[i] / self.agents_count
            for i in range(self.critic_layers_count):
                for j in range(self.agents_count):
                    critic_mean_weights[i] += self.agents[j].critic.layers[i].weight.data.clone()
                    critic_mean_bias[i] += self.agents[j].critic.layers[i].bias.data.clone()
                critic_mean_weights[i] = critic_mean_weights[i] / self.agents_count
                critic_mean_bias[i] = critic_mean_bias[i] / self.agents_count
        return Means(actor_mean_weights, actor_mean_bias, 
                     critic_mean_weights, critic_mean_bias)

    def agents_update_models(self, 
        means: tuple
        ) -> None:
        """Update parameter values of existing agents using computed means and soft udpate.

        Args:
            means (tuple): Averages
        """
        for i in range(self.agents_count):
            for j in range(self.actor_layers_count):
                self.agents[i].actor.layers[j].weight.data = \
                    (1 - self.TAU) * self.agents[i].actor.layers[j].weight.data + self.TAU * means.aw[j].clone()
                self.agents[i].actor.layers[j].bias.data = \
                    (1 - self.TAU) * self.agents[i].actor.layers[j].bias.data + self.TAU * means.ab[j].clone()
            for j in range(self.critic_layers_count):
                self.agents[i].critic.layers[j].weight.data = \
                    (1 - self.TAU) * self.agents[i].critic.layers[j].weight.data + self.TAU * means.cw[j].clone()
                self.agents[i].critic.layers[j].bias.data = \
                    (1 - self.TAU) * self.agents[i].critic.layers[j].bias.data + self.TAU * means.cb[j].clone()
        return
