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


class TreeLoader(object):
    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __iter__(self):
        return _DataLoaderIter(self)

    def __len__(self):
        return len(self.batch_sampler)