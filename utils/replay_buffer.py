"""A simple Replay Buffer using python list."""
import numpy as np
from scipy.special import softmax


class ReplayBuffer:

    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.data = list()
        self.index = 0
        self.length = 0

    def __len__(self):
        return self.length

    def add_paths(self, paths):
        if self.length < self.capacity:
            self.data.append(paths)
            self.index += 1
            self.length += 1
        else:
            self.data[self.index] = paths
            self.index += 1
        self.index %= self.capacity  # self.index in [0, self.capacity)

    def get_samples(self, batch_size, recent=False, alpha=0.1):
        """
        Get a batch of samples of from the attribute :data (list):. 

        recent (bool): if recent=True, use only recent samples.
        alpha (float): ratio of samples to use for "recent" sampling.
        
        return: a list of samples with length :batch_size:.
        """
        assert alpha > 0 and alpha <= 1, 'alpha out of range'
        assert self.can_sample(batch_size), 'do not have enough data'
        if not recent:
            sample_ids = np.random.choice(np.arange(self.length),
                                          size=batch_size,
                                          replace=False)
        else:  # Use only recent samples
            len_recent = int(alpha * self.length)
            if len_recent <= batch_size:
                idx_pool = np.arange(self.index - batch_size, self.index)
                weights = None
            else:
                idx_pool = np.arange(self.index - len_recent, self.index)
                weights = np.arange(len(idx_pool), dtype=np.float64)
                weights = softmax(weights)
            sample_ids = np.random.choice(idx_pool,
                                          size=batch_size,
                                          replace=False, p=weights)
        return [self.data[id] for id in sample_ids]
    
    def can_sample(self, size):
        return self.length >= size
