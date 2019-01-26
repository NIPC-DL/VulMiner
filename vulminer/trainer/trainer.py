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
    return batch


class Trainer(object):
    def __init__(self, root, dataset=None, device=dev):
        self._root = pathlib.Path(root).expanduser()
        self._root.mkdir(parents=True, exist_ok=True)
        self._device = device
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
            nn = model['nn'].to(self._device)
            opti = model['optimizer']
            loss = model['loss']
            batch_size = model['batch_size']
            epoch = model['epoch']
            if folds:
                for i in range(folds):
                    ts, vs = self._kfolds(folds)
                    train = DataLoader(self._dataset,
                            batch_size=batch_size, shuffle=False,
                            collate_fn=tree_collate, sampler=ts)
                    valid = DataLoader(self._dataset,
                            batch_size=batch_size, shuffle=False,
                            collate_fn=tree_collate, sampler=vs)
                    self._training(train, nn, opti, loss, epoch)
                    self._validing(valid, nn)

    def _training(self, train, nn, optimizer, loss, epoch):
        for i in range(epoch):
            logger.info(f'epoch {i+1} start')
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
                if idx % 50 == 0:
                    logger.info(f'loss: {tloss}')
                optimizer.step()
                optimizer.zero_grad()
            print(f'epoch {i+1} fininsed')
        return nn

    def _validing(self, valid, nn):
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in valid:
                for input, label in batch:
                    input = input.to(self._device)
                    label = label.to(self._device)
                    outputs = nn(input)
                    pred = torch.max(outputs.data, -1)
                    total += 1
                    correct += 1 if pred == label else 0
            print('Accuracy: {} %'.format(100 * correct / total)) 

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

    def _kfolds(self, folds):
        size = len(self._dataset)
        indices = list(range(size))
        np.random.shuffle(indices)
        split = int(np.floor(size / folds))
        train_idx, valid_idx = indices[:-split], indices[-split:]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        return train_sampler, valid_sampler
