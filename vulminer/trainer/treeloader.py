# -*- coding: utf-8 -*-
"""
treeloader.py - The Dataloader for Tree Structrue

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""
from torch.utils.data.sampler import RandomSampler, SequentialSampler, BatchSampler


class _DataLoaderIter:
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.batch_sampler = loader.batch_sampler
        self.sample_iter = iter(self.batch_sampler)

    def __len__(self):
        return len(self.batch_sampler)

    def __next__(self):
        indices = next(self.sample_iter)
        batch = [self.dataset[i] for i in indices]
        return batch


class TreeLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size

        self.sampler = RandomSampler(
            dataset) if shuffle else SequentialSampler(dataset)
        self.batch_sampler = BatchSampler(self.sampler, batch_size, False)

    def __iter__(self):
        return _DataLoaderIter(self)

    def __len__(self):
        return len(self.batch_sampler)