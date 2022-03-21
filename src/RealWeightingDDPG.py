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
import copy


Means = namedtuple('Averages', 'aw, ab, cw, cb')


class RealWeightingDDPG(IndividualDDPG):

    def __init__(self, 
        episode_count: int, 
        episode_step_count: int, 
        world: World
        ) -> None:
        self.NAME = 'RWDDPG'
        super().__init__(episode_count, episode_step_count, world)
        # get model coutns
        self.agents_count = len(self.agents)
        self.actor_layers_count = len(self.agents[0].actor.layers)
        self.critic_layers_count = len(self.agents[0].critic.layers)
        # averaging params
        self.TAU = 0.5
        # weights params
        self.BETA = 1
        self.MUL = 1
        self.ADD = 0.1
        self.agents_previous = copy.deepcopy(self.agents)
        self.agents_differences = copy.deepcopy(self.agents)
        return

    def agents_update(self,
        rewards: np.ndarray
        ) -> None:
        means = self.get_means(rewards)
        self.agents_update_models(means)
        return

    def get_means(self,
        rewards: np.ndarray
        ) -> tuple:
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
        # init standart deviations
        actor_std_weights = [None] * self.actor_layers_count
        actor_std_bias = [None] * self.actor_layers_count
        critic_std_weights = [None] * self.critic_layers_count
        critic_std_bias = [None] * self.critic_layers_count
        # compute differences and save them
        # compute standart deviations of differences
        with torch.no_grad():
            for i in range(self.actor_layers_count):
                actor_std_weights_l = []
                actor_std_bias_l = []
                for j in range(self.agents_count):
                    self.agents_differences[j].actor.layers[i].weight.data = self.agents_previous[j].actor.layers[i].weight.data - self.agents[j].actor.layers[i].weight.data
                    self.agents_differences[j].actor.layers[i].bias.data = self.agents_previous[j].actor.layers[i].bias.data - self.agents[j].actor.layers[i].bias.data
                    actor_std_weights_l.append(self.agents_differences[j].actor.layers[i].weight.data)
                    actor_std_bias_l.append(self.agents_differences[j].actor.layers[i].bias.data)
                actor_std_weights[i] = torch.std(torch.stack(actor_std_weights_l), dim=0)
                actor_std_bias[i] = torch.std(torch.stack(actor_std_bias_l), dim=0)
            for i in range(self.critic_layers_count):
                critic_std_weights_l = []
                critic_std_bias_l = []
                for j in range(self.agents_count):
                    self.agents_differences[j].critic.layers[i].weight.data = self.agents_previous[j].critic.layers[i].weight.data - self.agents[j].critic.layers[i].weight.data
                    self.agents_differences[j].critic.layers[i].bias.data = self.agents_previous[j].critic.layers[i].bias.data - self.agents[j].critic.layers[i].bias.data
                    critic_std_weights_l.append(self.agents_differences[j].critic.layers[i].weight.data)
                    critic_std_bias_l.append(self.agents_differences[j].critic.layers[i].bias.data)
                critic_std_weights[i] = torch.std(torch.stack(critic_std_weights_l), dim=0)
                critic_std_bias[i] = torch.std(torch.stack(critic_std_bias_l), dim=0)
        # compute weights
        # weights_sum = np.sum(np.abs(rewards) ** self.BETA)
        # weights = (np.sign(rewards) * (np.abs(rewards) ** self.BETA)) / weights_sum
        # weights_t = torch.from_numpy(weights).type(torch.cuda.FloatTensor)
        # compute means
        with torch.no_grad():
            for i in range(self.actor_layers_count):
                for j in range(self.agents_count):
                    actor_mean_weights[i] += rewards[j] * self.agents[j].actor.layers[i].weight.data.clone()
                    actor_mean_bias[i] += rewards[j] * self.agents[j].actor.layers[i].bias.data.clone()
                actor_mean_weights[i] = self.MUL * actor_mean_weights[i] / (self.agents_count * actor_std_weights[i] + self.ADD)
                actor_mean_bias[i] = self.MUL * actor_mean_bias[i] / (self.agents_count * actor_std_bias[i] + self.ADD)
            for i in range(self.critic_layers_count):
                for j in range(self.agents_count):
                    critic_mean_weights[i] += rewards[j] * self.agents[j].critic.layers[i].weight.data.clone()
                    critic_mean_bias[i] += rewards[j] * self.agents[j].critic.layers[i].bias.data.clone()
                critic_mean_weights[i] = self.MUL * critic_mean_weights[i] / (self.agents_count * critic_std_weights[i] + self.ADD)
                critic_mean_bias[i] = self.MUL * critic_mean_bias[i] / (self.agents_count * critic_std_bias[i] + self.ADD)
        return Means(actor_mean_weights, actor_mean_bias, 
                     critic_mean_weights, critic_mean_bias)

    def agents_update_models(self, 
        means: tuple
        ) -> None:
        for i in range(self.agents_count):
            for j in range(self.actor_layers_count):
                self.agents[i].actor.layers[j].weight.data = self.agents_previous[i].actor.layers[j].weight.data \
                    + self.TAU * self.agents_differences[i].actor.layers[j].weight.data + (1 - self.TAU) * means.aw[j]
                self.agents[i].actor.layers[j].bias.data = self.agents_previous[i].actor.layers[j].bias.data \
                    + self.TAU * self.agents_differences[i].actor.layers[j].bias.data + (1 - self.TAU) * means.ab[j].clone()
            for j in range(self.critic_layers_count):
                self.agents[i].critic.layers[j].weight.data = self.agents_previous[i].critic.layers[j].weight.data \
                    + self.TAU * self.agents_differences[i].critic.layers[j].weight.data + (1 - self.TAU) * means.cw[j].clone()
                self.agents[i].critic.layers[j].bias.data = self.agents_previous[i].critic.layers[j].bias.data \
                    + self.TAU * self.agents_differences[i].critic.layers[j].bias.data + (1 - self.TAU) * means.cb[j].clone()
        return
