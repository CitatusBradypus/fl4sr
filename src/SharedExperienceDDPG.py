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

class SharedExperienceDDPG(IndividualDDPG):
    """Shared Experience DDPG as described in theses.

    Args:
        IndividualDDPG (): ...
    """

    def __init__(self, 
        episode_count: int, 
        episode_step_count: int, 
        world: World
        ) -> None:
        """Initialize SEDDPG.

        Args:
            episode_count (int): ...
            episode_step_count (int): ...
            world (World): contains information about experiment characteristics
        """
        self.NAME = 'SEDDPG'
        self.BUFFER_SIZE = 50000
        super().__init__(episode_count, episode_step_count, world)
        return

    def init_buffers(self
        ) -> list:
        """Creates list with buffers.

        Returns:
            list: Buffers list.
        """
        return [BasicBuffer(self.BUFFER_SIZE)]

    def init_agents(self
        ) -> list:
        """Creates list with agents.

        Returns:
            list: Agents list.
        """
        return [DDPG(self.buffers[0], 
                     self.observation_dimension, 
                     self.action_dimension) 
                for i in range(self.robot_count)]

    def agents_actions(self,
        states: np.ndarray,
        ) -> np.ndarray:
        """Get actions for each agent. 

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
            self.buffers[0].add([Transition(s[i], a[i], r[i], s_[i], f[i])])
        return
