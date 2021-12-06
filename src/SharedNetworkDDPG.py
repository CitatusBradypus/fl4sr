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

    def __init__(self, 
        episode_count: int, 
        episode_step_count: int, 
        world: World
        ) -> None:
        super().__init__(episode_count, episode_step_count, world)
        # loggers
        self.NAME = 'SharedNetworkDDPG'
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
                     self.action_dimension)]

    def agents_actions(self,
        states: np.ndarray,
        ) -> np.ndarray:
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
        for i in range(self.robot_count):
            self.buffers[0].add([Transition(s[i], a[i], r[i], s_[i], f[i])])
        return
