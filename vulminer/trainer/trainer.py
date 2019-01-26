# -*- coding: utf-8 -*-
"""
mytrainer.py - description

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
    def __init__(self, root, dataset=None, device=dev):
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

    def fit(self, category=None, folds=None):
        for model in self._models:
            self.model_name = model['nn'].__class__.__name__
            nn = model['nn'].to(self._device)
            opti = model['optimizer']
            loss = model['loss']
            batch_size = model['batch_size']
            epoch = model['epoch']

            logger.info(f'model {self.model_name} start')
            if folds:
                for i in range(folds):
                    train, valid = self._dataset.load(category, folds)
                    tl = DataLoader(
                        train,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=tree_collate)
                    vl = DataLoader(
                        valid,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=tree_collate)
                    logger.info(f'folds [{i+1}/{folds}] start')
                    self._training(tl, nn, opti, loss, epoch)
                    self._validing(vl, nn)
                    logger.info(f'input size: {nn.input_size}')
                    logger.info(f'hidden size: {nn.hidden_size}')
                    logger.info(f'batch size: {batch_size}')
                    logger.info(f'epoch: {epoch}')
                    logger.info(f'lr: {opti.defaults["lr"]}')

    def _training(self, train, nn, opti, loss, epoch):
        for i in range(epoch):
            logger.info(f'epoch {i+1} start')
            for idx, (inputs, labels) in enumerate(train):
                tloss = 0.0
                inputs = inputs.to(dev)
                labels = labels.to(dev)
                opti.zero_grad()
                outputs = nn(inputs)
                err = loss(outputs, labels)
                opti.zero_grad()
                err.backward()
                opti.step()
                if idx % 100 == 0:
                    logger.info(f'loss: {tloss}')
            print(f'epoch {i+1} fininsed')
        return nn

    def _validing(self, valid, nn):
        with torch.no_grad():
            flag = True
            for batch in valid:
                for input, label in batch:
                    input = input.to(self._device)
                    label = label.to(self._device)
                    output = nn(input)
                    if flag:
                        o = output.expand(1, 2)
                        l = label.expand(1)
                        flag = False
                    else:
                        o = torch.cat((o, output.expand(1, 2)), 0)
                        l = torch.cat((l, label.expand(1)), 0)
            _, p = torch.max(o.data, 1)
            total = l.size(0)
            tp, tn, fp, fn = self._metrics(p, l.long())
            precision = 100 * tp / (tp + fp)
            fpr = 100 * fp / (fp + tn)
            fnr = 100 * fn / (fn + tp)
            f1 = 100 * 2 * tp / (2 * tp + fp + fn)
            logger.info(f'prc: {precision}')
            logger.info(f'fpr: {fpr}')
            logger.info(f'fnr: {fnr}')
            logger.info(f'f1: {f1}')

    def _metrics(self, pred, label):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for p, l in zip(pred, label):
            pi = p.item()
            li = l.item()
            if pi == 1 and li == 1:
                tp += 1
            elif pi == 1 and li == 0:
                fp += 1
            elif pi == 0 and li == 1:
                fn += 1
            else:
                tn += 1
        return tp, tn, fp, fn
