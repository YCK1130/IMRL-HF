#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import numpy as np
from typing import List


class Replay:
    def __init__(self, memory_size, batch_size, drop_prob=0, to_np=True):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = []
        self.pos = 0
        self.drop_prob = drop_prob
        self.to_np = to_np

    def feed(self, experience):
        if np.random.rand() < self.drop_prob:
            return
        if self.pos >= len(self.data):
            self.data.append(experience)
        else:
            self.data[self.pos] = experience
        self.pos = (self.pos + 1) % self.memory_size

    def feed_batch(self, experience):
        for exp in experience:
            self.feed(exp)

    def sample(self, batch_size=None):
        if self.empty():
            return None
        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = [np.random.randint(
            0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[ind] for ind in sampled_indices]
        sampled_data = zip(*sampled_data)
        if self.to_np:
            sampled_data = list(map(lambda x: np.asarray(x), sampled_data))
        return sampled_data

    def size(self):
        return len(self.data)

    def empty(self):
        return not len(self.data)

    def shuffle(self):
        np.random.shuffle(self.data)

    def clear(self):
        self.data = []
        self.pos = 0


class Storage:
    def __init__(self, size, keys: List[str]):
        self.keys = keys
        self.size = size
        self.reset()

    def add(self, data):
        for k, v in data.items():
            if k in self.keys:
                getattr(self, k).append(v)
            else:
                raise ValueError('Key %s not in storage' % k)

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.size)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def cat(self, keys):
        data = [getattr(self, k)[:self.size] for k in keys]
        return map(lambda x: torch.cat(x, dim=0), data)
