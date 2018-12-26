# -*- coding: utf-8 -*-
"""
mytrainer.py - description

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""
import torch
import pathlib
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from .treeloader import TreeLoader
from vulminer.utils import logger

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tree_collate(batch):
    pass


class Trainer(object):
    def __init__(self, root, dataset=None):
        self._root = pathlib.Path(root).expanduser()
        self._root.mkdir(parents=True, exist_ok=True)
        self._models = list()

    def addData(self, dataset, valid=None):
        if isinstance(dataset, Dataset):
            self._dataset = dataset
            if isinstance(valid, Dataset):
                self._valid = valid
        else:
            raise ValueError(f'{dataset} is not a torch Dataset object')

    def addModel(self, model):
        """Add models
        
        Args:
            model (list, dict): A model is a dict contained necessary values
                you can add it one by one or just add a list of models
            
        """
        if isinstance(model, list):
            self._models.extend(model)
        else:
            self._models.append(model)

    def fit(self, folds=None):
        for model in self._models:
            self.model_name = model['nn'].__class__.__name__
            self.nn = model['nn']
            self.opti = model['optimizer']
            self.loss = model['loss']
            self.batch_size = model['batch_size']
            self.epoch = model['epoch']

            if folds and isinstance(folds, int):
                for i in range(folds):
                    logger.info(f'Start [{i+1}/{folds}] fold')
                    train, valid = self._get_loader(folds)
                    nn = self._training(train, self.nn, self.opti, self.loss,
                                        self.epoch)
                    y_pred, y = self._testing(valid, nn)
                    with open(str(self._root / 'res.txt')) as f:
                        for yp, y in zip(y_pred, y):
                            f.write(str(yp) + ' ' + str(y) + '\n')
            else:
                train = TreeLoader(self._dataset, batch_size=self.batch_size)
                self._training(train, self.nn, self.opti, self.loss,
                               self.epoch)

    def _training(self, train, nn, optimizer, loss, epoch):
        for i in range(epoch):
            print(f'epoch {i+1} start')
            for idx, batch in enumerate(train):
                tloss = 0.0
                for input, label in batch:
                    input = input.to(dev)
                    label = label.to(dev)
                    optimizer.zero_grad()
                    output = nn(input)
                    err = loss(output, label)
                    err.backward()
                    tloss += err.item()
                if idx % 5 == 0:
                    logger.info(f'loss: {tloss}')
                optimizer.step()
                optimizer.zero_grad()
            print(f'epoch {i+1} fininsed')
        return nn

    def _testing(self, valid, nn):
        y_pred = []
        y = []
        for input, label in valid:
            input = input.to(dev)
            label = label.to(dev)
            output = nn(output)
            y_pred.append(torch.max(output, -1))
            y.append(torch.max(y, -1))
        return y_pred, y

    def _validation(self):
        pass

    def _get_loader(self, folds):
        """Get dataloader by given folds
        
        Args:
            folds (int): n-folds validation
        
        """
        size = len(self._dataset)
        indices = list(range(size))
        np.random.shuffle(indices)
        split = int(np.floor(size / folds))
        train_idx, valid_idx = indices[:-split], indices[-split:]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        if 'tnn' in self.model_name.lower():
            train_loader = TreeLoader(
                self._dataset,
                self.batch_size,
                sampler=train_sampler,
                shuffle=False)
            valid_loader = TreeLoader(
                self._dataset,
                self.batch_size,
                sampler=valid_sampler,
                shuffle=False)
        else:
            train_loader = DataLoader(
                self._dataset, sampler=train_sampler, shuffle=False)
            valid_loader = DataLoader(
                self._dataset, sampler=valid_sampler, shuffle=False)
        return train_loader, valid_loader
