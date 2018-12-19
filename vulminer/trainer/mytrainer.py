# -*- coding: utf-8 -*-
"""
mytrainer.py - description

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""
import torch
import pathlib
from torch.utils.data import Dataset, DataLoader
from .treeloader import TreeLoader

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    def fit(self):
        for model in self._models:
            train = TreeLoader(self._dataset, batch_size=50)
            self._training(train, model['nn'], model['optimizer'],
                           model['loss'], model['epoch'])

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
                    print(tloss)
                optimizer.step()
                optimizer.zero_grad()
            print(f'epoch {i+1} fininsed')

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
