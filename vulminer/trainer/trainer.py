# -*- coding: utf-8 -*-
"""
trainer.py - Trainer

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""
import torch
import pathlib
from torch.utils.data import DataLoader
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

    def fit(self, category=None, folds=None, save=False):
        """Start training and validation

        Args:
            category (list, None): The class of datasets
            folds (int): Folds of validation

        """
        for model in self._models:
            self.model_name = model['nn'].__class__.__name__
            nn = model['nn'].to(self._device)
            opti = model['opti']
            loss = model['loss']
            batch_size = model['batch_size']
            epoch = model['epoch']

            logger.info(f'model {self.model_name} start')
            if folds:
                for i in range(folds):
                    train, valid = self._dataset.load(category, folds)
                    tl = DataLoader(train, batch_size=batch_size, shuffle=False)
                    vl = DataLoader(valid, batch_size=batch_size, shuffle=False)
                    logger.info(f'folds [{i+1}/{folds}] start')
                    self._training(tl, nn, opti, loss, epoch)
                    self._validing(vl, nn)
                    if save:
                        torch.save(model.state_dict(),
                                   self._root / f'{self.model_name}.model')
                    break

    def _training(self, train, nn, opti, loss, epoch):
        for i in range(epoch):
            logger.info(f'epoch {i+1} start')
            for idx, (inputs, labels) in enumerate(train):
                inputs = inputs.to(dev)
                labels = labels.to(dev)

                outputs = nn(inputs)
                err = loss(outputs, labels)

                opti.zero_grad()
                err.backward()
                opti.step()

                if idx % 10 == 0:
                    logger.info(f'loss: {err.item()}')
            logger.info(f'epoch {i+1} fininsed')
        return nn

    def _validing(self, valid, nn):
        total_pred = []
        total_label = []
        with torch.no_grad():
            for inputs, labels in valid:
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)
                outputs = nn(inputs)
                _, p = torch.max(outputs.data, 1)
                total_pred.extend([int(x) for x in p.data])
                total_label.extend([int(x) for x in labels.data])
        for k, m in self._metrics.items():
            num = m(total_pred, total_label)
            logger.info(f'{k}: {num}')

    def predicts(self):
        pass
