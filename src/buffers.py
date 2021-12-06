#! /usr/bin/env python3.8
import sys
import os
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')
from collections import namedtuple
import random
import numpy as np


Transition = namedtuple('Transition', 's a r s_ f')
VectorTransitions = namedtuple('VectorTransitions', 's a r s_ f')


class BasicBuffer():
    """Implementation of simple circular buffer.
        (for some reason python doesn't have this one implemented)
    """
    
    def __init__(self, 
        max_size: int
        ) -> None:
        """Creates and already initialazes full circular buffer.

        Args:
            max_size (int): Buffer size.
        """
        self._values = [None] * max_size
        self._max_size = max_size
        self._index = 0
        self._values_count = 0

    def add(self, 
        transitions: list
        ) -> None:
        """Adds transitions to buffer.

        Args:
            transitions (list): Transitions to add.
        """
        for transition in transitions:
            self._values[self._index] = transition
            self._index = (self._index + 1) % self._max_size
            self._values_count = min(self._values_count + 1, self._max_size)

    def choice(self, 
        batch_size: int
        ) -> tuple:
        """Sample batch from buffer using random choice.

        Args:
            batch_size (int): Amount of sampled transitions.

        Returns:
            tuple: Named tuple of numpy arrays corresponding to transitions 
                    elements.
        """
        if self._values_count == self._max_size:
            samples = random.choices(self._values, 
                                     k=batch_size)
        else:
            samples = random.choices(self._values[:self._values_count], 
                                     k=batch_size)
        samples = vectorize_samples(samples)
        return samples

    def sample(self,
        batch_size: int
        ) -> tuple:
        """Sample batch from buffer using random sample.

            If buffer is not as large as batch_size method fails and returns 
            None.

        Args:
            batch_size (int): Amount of sampled transitions.

        Returns:
            tuple: Named tuple of numpy arrays corresponding to transitions 
                    elements.
        """
        if self._values_count < batch_size:
            return None
        if self._values_count == self._max_size:
            samples = random.sample(self._values, 
                                    k=batch_size)
        else:
            samples = random.sample(self._values[:self._values_count], 
                                    k=batch_size)
        samples = vectorize_samples(samples)
        return samples


def vectorize_samples(
    samples: list
    ) -> tuple:
    """Transforms samples to numpy arrays of values.

    Args:
        samples (list): List of transitions.

    Returns:
        tuple: Tuple of numpy arrays corresponding to elements of transition.
    """
    assert len(samples[0]) == 5, 'Wrong samples dimension!'

    size = len(samples)
    states, actions, rewards, states_, finished = [None] * size, [None] * size, [None] * size, [None] * size, [None] * size
    for i in range(len(samples)):
        states[i] = samples[i].s
        actions[i] = samples[i].a
        rewards[i] = samples[i].r
        states_[i] = samples[i].s_
        finished[i] = samples[i].f
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    states_ = np.array(states_)
    finished = np.array(finished)
    return VectorTransitions(states, actions, rewards, states_, finished)


if __name__ == '__main__':

    buffer = BasicBuffer(2)
    
    transitions = []
    for i in range(4):
        transitions.append(Transition(np.random.normal(size=(10)),
                                      np.random.normal(size=(2)),
                                      np.random.normal(size=(1)),
                                      np.random.normal(size=(10)),
                                      np.random.randint(2, size=(1))))
        print(transitions[i])

    buffer.add(transitions)
    print('buffer')
    print(buffer._values[0])
    print(buffer._values[1])
    
    sample = buffer.sample(2)
    print('sample')
    print(sample)
