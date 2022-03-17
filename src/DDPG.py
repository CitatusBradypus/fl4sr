#! /usr/bin/env python3.8
import sys
import os
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')
from models import Actor, Critic
from buffers import BasicBuffer, PrioritizedExperienceReplayBuffer, VectorTransitions, Transition
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DDPG:
    """Deep deterministic policy gradient.
    """
        
    def __init__(self,
        replay_buffer,
        state_dimension: int,
        action_dimension: int,
        ) -> None:
        """Creates actor, critic and target actor and critic, sets their weights 
            to same values.

        Args
            replay_buffer (Buffer): Buffer holding transition samples.
            state_dimension (int): Dimension of state observations.
            action_dimension (int): Dimension of robot actions.
        """
        # store parameters
        self.state_dimension = state_dimension
        self.action_dimension = action_dimension
        self.replay_buffer = replay_buffer
        # other parameters
        # actor, critic parameters
        self.ACTOR_HIDDEN_LAYERS = [64, 64, 64]
        self.CRITIC_HIDDEN_LAYERS = [64, 64, 64]
        # training parameters
        self.LEARNING_RATE_ACTOR = 0.001
        self.LEARNING_RATE_CRITIC = 0.001
        self.BATCH_SIZE = 512
        self.GAMMA = 0.99
        # update parameters
        self.TAU = 0.5
        # actor networks init and cuda
        self.actor = Actor(self.state_dimension, 
                           self.ACTOR_HIDDEN_LAYERS)
        self.actor.cuda()
        self.actor_target = Actor(self.state_dimension, 
                                  self.ACTOR_HIDDEN_LAYERS)
        self.actor_target.cuda()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), 
                                          self.LEARNING_RATE_ACTOR)
        update_parameters(self.actor_target, self.actor)
        # critic networks init and cuda
        self.critic = Critic(self.state_dimension,
                             self.action_dimension, 
                             self.CRITIC_HIDDEN_LAYERS)
        self.critic.cuda()
        self.critic_target = Critic(self.state_dimension,
                                    self.action_dimension, 
                                    self.CRITIC_HIDDEN_LAYERS)
        self.critic_target.cuda()
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           self.LEARNING_RATE_CRITIC)
        update_parameters(self.critic_target, self.critic)
        return

    def select_action(self, 
        state: np.ndarray
        ) -> np.ndarray:
        """Runs actor neural net with state as input, returns resulting action
            as numpy array.

        Args:
            state (np.ndarray): Input state.

        Returns:
            np.ndarray: 
        """
        # state from numpy to cuda tensor (also sends tensor to cuda)
        x = torch.from_numpy(state.reshape(1, -1)).type(torch.cuda.FloatTensor)
        # run actor nn
        action = self.actor(x)
        # move result to cpu, detach, convert to numpy array
        action = action.cpu().detach().numpy()[0]
        return action
    
    def train(self
        ) -> None:
        """Train actor and critic neural networks.
        """
        # check amount of samples in buffer
        if self.replay_buffer._values_count < self.BATCH_SIZE:
            return
        # sample transitions
        if isinstance(self.replay_buffer, PrioritizedExperienceReplayBuffer):
            transitions, weights = self.replay_buffer.sample(self.BATCH_SIZE)
            weights = np.sqrt(weights)
            weights_t = torch.from_numpy(weights).type(torch.cuda.FloatTensor)
        else:
            transitions = self.replay_buffer.sample(self.BATCH_SIZE)
        states, actions, rewards, states_next, finished = transitions
        # to tensors
        states_t = torch.from_numpy(states).type(torch.cuda.FloatTensor)
        actions_t = torch.from_numpy(actions).type(torch.cuda.FloatTensor)
        rewards_t = torch.from_numpy(rewards.reshape(-1, 1)).type(torch.cuda.FloatTensor)
        states_next_t = torch.from_numpy(states_next).type(torch.cuda.FloatTensor)
        finished_t = torch.from_numpy(finished.reshape(-1, 1)).type(torch.cuda.FloatTensor)
        # prepare condensed inputs
        states_actions_t = (states_t, actions_t)
        # get target actions from next state
        actions_target_t = self.actor_target(states_next_t)
        states_actions_next_t = (states_next_t, actions_target_t)
        # critic update
        self.critic.zero_grad()
        q_current_t = self.critic(states_actions_t)
        q_target_next_t = self.critic_target(states_actions_next_t)
        q_target_t = rewards_t + finished_t * self.GAMMA * q_target_next_t
        criterion = nn.MSELoss()
        if isinstance(self.replay_buffer, PrioritizedExperienceReplayBuffer):
            q_difference_t = q_target_t - q_current_t
            q_difference_weighted_t = torch.mul(q_difference_t, weights_t)
            zeros_t = torch.zeros(q_difference_weighted_t.shape).cuda()
            critic_loss = criterion(q_difference_weighted_t, zeros_t)
            self.replay_buffer.update(q_difference_t.detach().cpu().numpy())
        else:
            critic_loss = criterion(q_target_t, q_current_t)
        critic_loss.backward()
        self.critic_optimizer.step()
        # actor update
        self.actor.zero_grad()
        actions_actor_t = self.actor(states_t)
        states_actions_actor_t = (states_t, actions_actor_t)
        policy_loss = - self.critic(states_actions_actor_t)
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()
        return

    def update_targets(self,
        hard: bool=False
        ) -> None:
        """Update actor and critic target neural networks using predefined 
            parameter TAU.
        """
        if hard:
            update_parameters(self.actor_target, self.actor)
            update_parameters(self.critic_target, self.critic)
        else:
            update_parameters(self.actor_target, self.actor, self.TAU)
            update_parameters(self.critic_target, self.critic, self.TAU)
        return

    def weights_save(self, 
        path_actor: str, 
        path_critic: str
        ) -> None:
        torch.save(self.actor.state_dict(),
                   path_actor)
        torch.save(self.critic.state_dict(),
                   path_critic)
        return

    def weights_load(self, 
        path_actor: str, 
        path_critic: str
        ) -> None:
        self.actor.load_state_dict(torch.load(path_actor))
        self.critic.load_state_dict(torch.load(path_critic))
        return

def update_parameters(
    target: nn.Module,
    source: nn.Module,
    tau: float=0
    ) -> None:
    """Soft updates parameters in target nn.Model using source parameters. 
        If tau is not specified update defaults to hard.

    Args:
        target (nn.Module): Update destination
        source (nn.Module): Update source
        tau (float, optional): Update constant for soft weight update. 
            Defaults to 0.
    """
    for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
        target_parameter.data.copy_((1 - tau) * target_parameter + tau * source_parameter)
    return


if __name__ == '__main__':
    # buffer preparation
    buffer = BasicBuffer(256)
    transitions = []
    for i in range(64):
        # s a r s_ f
        transitions.append(Transition(np.random.normal(size=(10)),
                                      np.random.normal(size=(2)),
                                      np.random.normal(size=(1)),
                                      np.random.normal(size=(10)),
                                      np.random.randint(2, size=(1))))
    buffer.add(transitions)
    # ddpg creation
    ddpg = DDPG(buffer, 10, 2)
    # test select_action
    print("TEST select_action")
    state = np.random.normal(size=(10))
    action = ddpg.select_action(state)
    print('a0: ', action)
    # test train
    print("TEST train")
    BATCH_SIZE = 32
    ddpg.train()
    action = ddpg.select_action(state)
    print('a1: ', action)
