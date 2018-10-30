# -*- coding: utf-8 -*-
"""
trainer.py - 

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""
import os
import torch
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy, Precision, Recall, Loss
from vulminer.utils import logger
from vulminer.utils import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, root):
        self._root = utils.ensure_path_exist(root)
        self._models = []
        self._metrics = {}
        self._events = {
            'started': [],
            'completed': [],
            'epoch_completed': [],
            'epoch_started': [],
            'exception_raised': [],
            'iteration_completed': [],
            'iteration_started': []
        }

    def addData(self, train, valid=None):
        self._train = train
        self._valid = valid

    def addMetrics(self, metric):
        self._metrics.update(metric)

    def addModel(self, model):
        if isinstance(model, list):
            self._models.extend(model)
        else:
            self._models.append(model)

    def fix(self, save=True):
        path = utils.ensure_path_exist(self._root + 'Trained_Models/')
        if save:
            path = os.path.expanduser(path)
            if not os.path.exists(path):
                os.makedirs(path)
        for model in self._models:
            nn = self._train(model)
            torch.save(nn.state_dict(), f'{nn.__class__.__name__}_nn.pt')

    def addEvent(self, event_name, event_handler):
        self._events[event_name].append(event_handler)

    def _training(self, model):
        nn = model['nn']
        opt = model['optimizer']
        l = model['loss']
        epoch = model['epoch']
        self.addMetrics({'loss': Loss(l)})
        trainer = create_supervised_trainer(nn, opt, l, device=device)
        evaluator = create_supervised_evaluator(
            nn, metrics=self._metrics, device=device)
        # add event handler to trainer
        for event_name, event_handler in self._events.items():
            for i in event_handler:
                trainer.add_event_handler(k, i, evaluator, self._train,
                                          self._valid)

        @trainer.on(Events.EPOCH_COMPLETED)
        def train_result(trainer):
            evaluator.run(self._train)
            metrics = evaluator.state.metrics
            logger.info(
                f"Train Result - Epoch: {trainer.state.epoch} Loss: {metric['loss']}"
            )

        @trainer.on(Events.EPOCH_COMPLETED)
        def valid_result(trainer):
            if self._valid:
                evaluator.run(self._valid)
                if trainer.state.epoch == epoch:
                    pass

        trainer.run(self._train, max_epochs=epoch)
        return nn