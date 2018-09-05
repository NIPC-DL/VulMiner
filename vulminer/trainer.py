#!/usr/bin/env python3
#coding: utf-8

import os
import numpy as np
from logger import logger
from models import BGRU
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from configer import configer
from dataset import VulDataset
import utils
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy, Precision, Recall, Loss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _get_data(kf):
    dataset = np.load('../Cache/dataset0.npz')
    dsize = os.path.getsize('../Cache/dataset0.npz')
    x_set, y_set = dataset['arr_0'], dataset['arr_1']
    rng = np.arange(x_set.shape[0])
    np.random.shuffle(rng)
    x_set_r = []
    y_set_r = []
    for i in rng:
        x_set_r.append(x_set[i])
        y_set_r.append(y_set[i])
    x_set_r = np.asarray(x_set_r)
    y_set_r = np.asarray(y_set_r)
    pers = round((1.0/kf) * len(x_set_r))
    return (x_set_r[pers:], y_set_r[pers:]), (x_set_r[:pers], y_set_r[:pers])

def _get_data_loader(kf, batch_size):
    train, valid= _get_data(kf)
    train_loader = DataLoader(dataset=VulDataset(train), batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(dataset=VulDataset(valid), batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader

class Trainer_():
    def __init__(self):
        config = configer.getModel()
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.num_classes = config['num_classes']
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.dropout = config['dropout']
        self.learning_rate = config['learning_rate']

        self.res = {'acc':[], 'loss': [], 'prec': [], 'recall': [], 'f1': []}

    def folds(self, kf):
        model = BGRU(self.input_size, self.hidden_size, self.num_layers, self.num_classes, self.batch_size, self.dropout)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        train_loader, valid_loader = _get_data_loader(kf, self.batch_size)
        trainer = create_supervised_trainer(model, optimizer, loss, device=DEVICE)
        evaluator = create_supervised_evaluator(
                model,
                metrics={
                    'acc': CategoricalAccuracy(),
                    'loss': Loss(loss),
                    'prec': Precision(average=True),
                    'recall': Recall(average=True)
                    },
                device=DEVICE
                )

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(trainer):
            iter_num = trainer.state.iteration
            if iter_num%10 == 0:
                logger.info("Epoch[{}] Iter: {} Loss: {:.2f}".format(trainer.state.epoch, iter_num, trainer.state.output))

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics
            f1 = (2*metrics['prec']*metrics['recall'])/(metrics['prec'] + metrics['recall'])
            logger.info("Train Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f} Avg Precision: {:.2f} Avg Recall: {:.2f} Avg F1 Score: {:.2f}"
                  .format(trainer.state.epoch, metrics['acc'], metrics['loss'], metrics['prec'], metrics['recall'], f1))

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            evaluator.run(valid_loader)
            metrics = evaluator.state.metrics
            f1 = (2*metrics['prec']*metrics['recall'])/(metrics['prec'] + metrics['recall'])
            for k in self.res.keys():
                if k != 'f1':
                    self.res[k].append(metrics[k])
                else:
                    self.res[k].append(f1)
            logger.info("Valid Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f} Avg Precision: {:.2f} Avg Recall: {:.2f} Avg F1 Score: {:.2f}"
                  .format(trainer.state.epoch, metrics['acc'], metrics['loss'], metrics['prec'], metrics['recall'], f1))

        trainer.run(train_loader, max_epochs=self.num_epochs)
        return model

    def fit(self, kf):
        for i in range(kf):
            logger.info("[{}|{}] Folds Start".format(str(i+1), str(kf)))
            model = self.folds(kf)
            torch.save(model.state_dict(), 'model{}.pt'.format(i))
        f = open('result.txt', 'a')
        for k, v in self.res.items():
            f.write("{0}: {1}\n".format(k, sum(v)/len(v)))
            print("{}: {:.2f}".format(k, sum(v)/len(v)))
        f.close()

