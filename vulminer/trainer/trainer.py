# -*- coding: utf-8 -*-
"""
trainer.py - Trainer

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""
import torch
import pathlib
from vulminer.utils import logger

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer(object):
    def __init__(self, root, device=dev, padded=True):
        self._root = pathlib.Path(root).expanduser()
        self._root.mkdir(parents=True, exist_ok=True)
        self._device = device
        self._models = list()
        self._padded = padded

    def addData(self, train, valid=None):
        self._train = train
        self._valid = valid

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

    def fit(self):
        """Start training and validation"""
        for model in self._models:
            mod = model['nn']
            mod_name = mod.__class__.__name__
            mod.to(self._device)
            opti = model['opti']
            crit = model['crit']
            epochs = model['epochs']

            logger.info(f'model {mod_name} training start')
            mod = self._training(mod, crit, opti, epochs)
            return mod

    def _training(self, mod, crit, opti, epochs):
        total_step = len(self._train)
        for epoch in range(epochs):
            for ind, (datas, lens, labels) in enumerate(self._train):
                datas = datas.to(self._device)
                lens = lens.to(self._device)
                labels = labels.to(self._device)
                if self._padded:
                    lens, perm_idx = lens.sort(0, descending=True)
                    datas = datas[perm_idx]
                    labels = labels[perm_idx]
                    outputs = mod(datas, lens)
                else:
                    outputs = mod(datas)
                loss = crit(outputs, labels)
                opti.zero_grad()
                loss.backward()
                opti.step()

                if (ind+1) % 200 == 0:
                    logger.info(
                            f'e [{epoch+1}/{epochs}] s [{ind+1}/{total_step}] l {loss.item():.4f}'
                            )
            if (epoch+1) > 3 and self._valid:
                self._validing(mod, crit)
        return mod

    def _validing(self, mod, crit):
        with torch.no_grad():
            total_pred = []
            total_label = []
            total_loss = []
            for ind, (datas, lens, labels) in enumerate(self._valid):
                datas = datas.to(self._device)
                lens = lens.to(self._device)
                labels = labels.to(self._device)
                if self._padded:
                    lens, perm_idx = lens.sort(0, descending=True)
                    datas = datas[perm_idx]
                    labels = labels[perm_idx]
                    outputs = mod(datas, lens)
                else:
                    outputs = mod(datas)
                loss = crit(outputs, labels)

                total_loss.append(loss.item())
                _, pred = torch.max(outputs.data, 1)
                total_pred.extend([int(x) for x in pred.data])
                total_label.extend([int(x) for x in labels.data])
        logger.info(f'average loss: {sum(total_loss)/len(total_loss)}')
        for k, m in self._metrics.items():
            logger.info(f'{k}: {m(total_label, total_pred)}')
        return total_pred

def predictor(mod, data, metrics=None, labels=True, padded=True, device=dev):
    logger.info('start predicting')
    with torch.no_grad():
        total_pred = []
        total_label = []
        for ind, samp in enumerate(data):
            x = samp[0].to(device)
            l = samp[1].to(device)
            if labels:
                y = samp[2].to(device)
            if padded:
                l, perm_idx = l.sort(0, descending=True)
                x = x[perm_idx]
                if labels:
                    y = y[perm_idx]
                outputs = mod(x, l)
            else:
                outputs = mod(x)
            _, pred = torch.max(outputs.data, 1)
            total_pred.extend([int(x) for x in pred.data])
            if labels:
                total_label.extend([int(x) for x in y.data])
    if metrics is not None and labels:
        for k, m in metrics.items():
            logger.info(f'{k}: {m(total_label, total_pred)}')
    return total_pred

