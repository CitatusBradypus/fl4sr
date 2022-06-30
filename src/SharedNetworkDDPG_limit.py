#! /usr/bin/env python3.8
import sys
import os
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')
from IndividualDDPG import IndividualDDPG
from worlds import World
from DDPG import DDPG
from buffers import BasicBuffer, Transition
import numpy as np
from models import Actor, Critic
from buffers import BasicBuffer, PrioritizedExperienceReplayBuffer, VectorTransitions, Transition
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SharedNetworkDDPG(IndividualDDPG):
    """Shared Network DDPG as described in thesis.

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
        """Initialize SNDDPG.

        Args:
            episode_count (int): ...
            episode_step_count (int): ...
            world (World): contains information about experiment characteristics
        """
        self.NAME = 'SNDDPG'
        self.BUFFER_TMP_SIZE = 70000
        self.BUFFER_SIZE = 10000
        super().__init__(episode_count, episode_step_count, world, env, reward_goal, reward_collision, reward_progress, reward_max_collision, list_reward, factor_linear, factor_angular, discount_factor, is_progress, name)
        return

    def init_buffers(self
        ) -> list:
        """Creates list with buffers.

        Returns:
            list: Buffers list.
        """
        return [BasicBuffer(self.BUFFER_SIZE) 
                for i in range(self.robot_count)] + [BasicBuffer(self.BUFFER_TMP_SIZE)]

    def init_agents(self
        ) -> list:
        """Creates list with agents.

        Returns:
            list: Agents list.
        """
        return [DDPG(self.buffers[i], 
                     self.observation_dimension, 
                     self.action_dimension,
                     self.discount_factor) 
                for i in range(self.robot_count)] + [DDPG(self.buffers[self.robot_count],  self.observation_dimension, 
                     self.action_dimension,
                     self.discount_factor)]

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
        print(f"self.env: {self.env}")
        # epizode loop
        for episode in range(self.episode_error, self.episode_count):
            self.enviroment.reset()
            print(f"environment reset.")
            current_states = self.enviroment.get_current_states()
            data_total_rewards = np.zeros(self.robot_count)
            if self.episode_error != episode:
                self.episode_step_error = 0
            for step in range(self.episode_step_error, self.episode_step_count):
                # get actions
                #print("got action")
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

    def agents_actions(self,
        states: np.ndarray,
        ) -> np.ndarray:
        """Get actions for each agent from one network.

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
        """Save all transitions to one shared buffer.
        Arguments are named as described in thesis, only "D" is renamed to "f".

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

    def agents_train(self):
        """Train all agents.
        """
        self.agents[-1].train() 
        return

    def agents_target(self):
        """Update target networks of all agents.
        """
        self.agents[-1].update_targets() 

    def agents_update(self, rewards):
        """Update parameters of agents.
        (Obviously empty for IDDPG, SEDDPG, and SNDDPG)

        Args:
            rewards (np.array): average rewards obtained by agents between updates 
        """
        self.aggregate_buffer()
        self.distribute_network()
        return
    

    def aggregate_buffer(self
        )-> None:
        """Aggregate all the samples from the individual buffer into a central buffer.

        """
        for i in range(self.robot_count):
            size_buffer = self.buffers[i]._values_count
            samples_buffer = self.buffers[i].pure_sample(int(size_buffer))
            if not (samples_buffer == None):
                self.buffers[self.robot_count].add(samples_buffer)
        

    def distribute_network(self
        )-> None:
        """Copy the central network and distribute to the individual network.

        """
        for i in range(self.robot_count):
            self.update_network(self.agents[i], self.agents[self.robot_count], hard=True)

    def update_network(self,
        agent_target: DDPG,
        agent_source: DDPG,
        hard: bool=False
        ) -> None:
        """Update actor and critic target neural networks using predefined 
            parameter TAU.
        """
        if hard:
            update_parameters(agent_target.actor, agent_source.actor)
            update_parameters(agent_target.actor_target, agent_source.actor_target)
            update_parameters(agent_target.critic, agent_source.critic)
            update_parameters(agent_target.critic_target, agent_source.critic_target)
        else:
            update_parameters(agent_target.actor, agent_source.actor + agent_source.RHO)
            update_parameters(agent_target.actor_target, agent_source.actor_target + agent_source.RHO)
            update_parameters(agent_target.critic, agent_source.critic + agent_source.RHO)
            update_parameters(agent_target.critic_target, agent_source.critic_target + agent_source.RHO)
        return

def update_parameters(
    target: nn.Module,
    source: nn.Module,
    rho: float=1
    ) -> None:
    """Soft updates parameters in target nn.Model using source parameters. 
        If tau is not specified update defaults to hard.

    Args:
        target (nn.Module): Update destination
        source (nn.Module): Update source
        tau (float, optional): Update constant for soft weight update. 
            Defaults to 1.
    """
    for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
        target_parameter.data.copy_((1 - rho) * target_parameter + rho * source_parameter)
    return