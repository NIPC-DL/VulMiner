# -*- coding: utf-8 -*-
"""
trainer.py - Trainer

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""
import torch
import pathlib
import pickle
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from .treeloader import TreeLoader
from vulminer.utils import logger

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer(object):
    def __init__(self, root, device=dev):
        self._root = pathlib.Path(root).expanduser()
        self._root.mkdir(parents=True, exist_ok=True)
        self._device = device
        self._models = list()

    def addData(self, dataset):
        self._dataset = dataset

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

    def addMetrics(self, metrics):
        self._metrics = metrics

    def fit(self, category=None, folds=None):
        """Start training and validation

        Args:
            category (list, None): The class of datasets
            folds (int): Folds of validation

        """
        for model in self._models:
            self.model_name = model['nn'].__class__.__name__
            mod = model['nn']
            mod.to(self._device)
            opti = model['opti']
            loss = model['loss']
            batch_size = model['batch_size']
            epoch = model['epoch']

            logger.info(f'model {self.model_name} start')
            if folds:
                for i in range(folds):
                    train, valid = self._dataset.load(category, folds)
                    print('load dataset success')
                    tl = DataLoader(
                        train,
                        batch_size=batch_size,
                        shuffle=True)
                    vl = DataLoader(
                        valid,
                        batch_size=batch_size,
                        shuffle=False)
                    logger.info(f'folds [{i+1}/{folds}] start')
                    self._training(tl, mod, opti, loss, epoch, valid=vl)
                    # self._validing(vl, mod)
                    break

    def _training(self, train, mod, opti, loss, epoch, valid=None):
        for i in range(epoch):
            logger.info(f'epoch {i+1} start')
            for idx, (inputs, labels) in enumerate(train):
                inputs = inputs.to(dev)
                labels = labels.to(dev)
                # lengths = torch.LongTensor([len(x) for x in inputs]).to(dev)
                # outputs = mod(inputs, lengths)
                outputs = mod(inputs)
                err = loss(outputs, labels)
                opti.zero_grad()
                err.backward()
                opti.step()

                if idx % 200 == 0:
                    logger.info(f'Epoch [{i+1}/{epoch}] \
                            Step [{idx}/{len(train)}] \
                            loss: {err.item():.4f}')
            if valid and (i+1)%2 == 0:
                self._validing(valid, mod)
            logger.info(f'epoch {i+1} fininsed')

    def _validing(self, valid, mod):
        total_pred = []
        total_label = []
        with torch.no_grad():
            flag = True
            for inputs, labels in valid:
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)
                # lengths = torch.LongTensor([len(x) for x in inputs]).to(dev)
                outputs = mod(inputs)
                _, p = torch.max(outputs.data, 1)
                _, l = torch.max(labels.data, 1)
                total_pred.extend([int(x) for x in p.data])
                total_label.extend([int(x) for x in l.data])
        for k, m in self._metrics.items():
            num = m(total_pred, total_label)
            logger.info(f'{k}: {num}')

