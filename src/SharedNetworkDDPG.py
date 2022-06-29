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
        self.BUFFER_SIZE = 70000
        super().__init__(episode_count, episode_step_count, world, env, reward_goal, reward_collision, reward_progress, reward_max_collision, factor_linear, factor_angular, discount_factor, is_progress, name)
        return

    def init_buffers(self
        ) -> list:
        """Creates list with buffers.

        Returns:
            list: Buffers list.
        """
        return [self.BUFFER_TYPE(self.BUFFER_SIZE)]

    def init_agents(self
        ) -> list:
        """Creates list with agents.

        Returns:
            list: Agents list.
        """
        return [DDPG(self.buffers[0], 
                     self.observation_dimension, 
                     self.action_dimension,
                     self.discount_factor)]

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
            actions.append(self.agents[0].select_action(states[i]))
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
            self.buffers[0].add([Transition(s[i], a[i], r[i], s_[i], f[i])])
        return
