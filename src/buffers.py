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
        """Creates and already initializes full circular buffer.

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


class PrioritizedExperienceReplayBuffer():
    """Implementation of prioritized experience replay buffer.
    Implemented without sum tree, therefore it's realy slow.
    It should not be used, ever.
    """
    
    def __init__(self, 
        max_size: int
        ) -> None:
        """Creates and already initialazes experience replay buffer.

        Args:
            max_size (int): Buffer size.
        """
        # default circular buffer
        self._values = [None] * max_size
        self._max_size = max_size
        self._index = 0
        self._values_count = 0
        # PER
        # parameters
        self._alpha = 0.6
        self._beta = 0.4
        self._beta_delta = 0.0000234
        # arrays
        self._fresh_array = np.ndarray([0, 2])
        self._sorted_array = np.ndarray([0, 2])
        # other helpers
        self._full = False
        return

    def add(self, 
        transitions: list
        ) -> None:
        """Adds transitions to PER buffer.

        Args:
            transitions (list): Transitions to add.
        """
        new_indexes = []
        
        for transition in transitions:
            new_indexes.append(self._index)
            self._values_count = min(self._values_count + 1, self._max_size)
            self._values[self._index] = transition
            self._index = (self._index + 1) % self._max_size
        
        new_indexes_array = np.array(new_indexes)
        # search for alredy fresh and sorted indexes from new indexes
        already_fresh = np.isin(new_indexes_array, self._fresh_array[:, 1])
        already_sorted = np.isin(self._sorted_array[:, 1], new_indexes_array[~already_fresh])
        # delete new indexes in randked
        self._sorted_array = np.delete(self._sorted_array, already_sorted, axis=0)
        # create new fresh indexes and add them
        new_fresh = np.array((np.zeros(len(new_indexes_array[~already_fresh])), 
                             new_indexes_array[~already_fresh])).T
        self._fresh_array = np.concatenate((self._fresh_array, new_fresh))
        return

    def sample(self,
        batch_size: int
        ) -> tuple:
        """Sample batch from buffer according to PER rules.
        If buffer is not as large as batch_size method fails and returns None.

        Args:
            batch_size (int): Amount of sampled transitions.

        Returns:
            tuple: Named tuple of numpy arrays corresponding to transitions 
                    elements.
        """
        if self._values_count < batch_size:
            return None
        fresh_length = len(self._fresh_array)
        sorted_length = len(self._sorted_array)
        # combine fresh and sorted
        sorted_indexes = np.argsort(-self._sorted_array[:, 0])
        self._sorted_array = self._sorted_array[sorted_indexes]
        ranked_array = np.concatenate((np.ones(fresh_length),
                                       1 + np.arange(sorted_length)))
        # compute probabilities
        d_alpha_array = 1 / ranked_array ** self._alpha
        d_alpha_sum = np.sum(d_alpha_array)
        probability_array = d_alpha_array / d_alpha_sum
        # sample
        sample_indexes = np.random.choice(fresh_length + sorted_length, 
                                          size=batch_size, 
                                          replace=False, 
                                          p=probability_array)
        self._sample_fresh_indexes = sample_indexes[sample_indexes < fresh_length]
        self._sample_sorted_indexes = sample_indexes[sample_indexes >= fresh_length] - fresh_length
        sample_value_indexes = np.concatenate((self._fresh_array[self._sample_fresh_indexes, 1], 
                                               self._sorted_array[self._sample_sorted_indexes, 1]))
        samples = [self._values[s] for s in sample_value_indexes.astype(int)]
        samples = vectorize_samples(samples)
        weights = 1 / ((self._values_count * probability_array) ** self._beta)
        weights = weights[sample_indexes]
        return samples, weights

    def update(self,
        td_errors: np.ndarray
        ) -> None:
        """Updated PER buffer values.

        Args:
            td_errors (np.ndarray): errors of selected samples
        """
        assert len(self._sample_fresh_indexes) + len(self._sample_sorted_indexes) == len(td_errors), 'ERROR: PER buffer wrong update dimensions!'
        # set td errors to fresh indexes
        fresh_pairs = self._fresh_array[self._sample_fresh_indexes, :]
        fresh_pairs[:, 0] = td_errors[0:len(fresh_pairs)][:, 0]
        # remove used indexes from fresh
        self._fresh_array = np.delete(self._fresh_array, self._sample_fresh_indexes, axis=0)
        # set td error to sorted indexes
        self._sorted_array[self._sample_sorted_indexes, 0] = td_errors[len(fresh_pairs):][:, 0]
        # join sorted and used indexes
        self._sorted_array = np.concatenate((self._sorted_array, 
                                             fresh_pairs))
        self._beta = min(self._beta + self._beta_delta, 1)
        return


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

    buffer = PrioritizedExperienceReplayBuffer(4)
    #buffer._sorted_array = np.concatenate((buffer._sorted_array, np.array([[1, 3]])))

    transitions = []
    for i in range(4):
        transitions.append(Transition(np.random.normal(size=(10)),
                                      np.random.normal(size=(2)),
                                      np.random.normal(size=(1)),
                                      np.random.normal(size=(10)),
                                      np.random.randint(2, size=(1))))
        print(transitions[-1])
    buffer.add(transitions)
    samples, weights = buffer.sample(2)
    buffer.update(np.array([5.23, 1.24]))
    print('F =', buffer._fresh_array)
    print('S =', buffer._sorted_array)
    samples, weights = buffer.sample(2)
    buffer.update(np.array([2.24, 4.23]))
    print('F =', buffer._fresh_array)
    print('S =', buffer._sorted_array)
    #buffer.add(transitions)
    #buffer.add(transitions)
    
