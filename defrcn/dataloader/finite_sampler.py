import itertools
from typing import Optional

import torch
from detectron2.data.samplers import TrainingSampler
from detectron2.utils import comm
from torch.utils.data import Sampler


class FiniteTrainingSampler(Sampler):
    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = None):
        """
        Notes:
            This is the same as det2's TrainingSampler, but the indices are not infinite. Used for going through the dataset
            for a single pass
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        if self._shuffle:
            indices = torch.randperm(self._size, generator=g).tolist()
        else:
            indices = list(range(self._size))
        yield from indices
