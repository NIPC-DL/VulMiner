#!/usr/bin/env python3
#coding: utf-8

import numpy as np
from logger import logger
from models import BGRU
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from configer import configer
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer(object):
    def __init__(self):
        pass

    def init(self):
        config = configer.getModel()
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.num_classes = config['num_classes']
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.learning_rate = config['learning_rate']
        self.model = BGRU(self.input_size, self.hidden_size, self.num_layers, self.num_classes, self.batch_size).to(device)

    def fit(self, train_dataset):
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        total_step = len(self.train_loader)
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.num_epochs):
            for i, (vulvector, labels) in enumerate(self.train_loader):
                model.hidden = model.init_hidden()
                # load data
                vulvector = torch.tensor(vulvector).float().to(device)
                labels = labels.long().to(device) # Forward pass
                # get output
                outputs = self.model(vulvector)
                # caculate loss
                loss = criterion(outputs, labels)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, self.num_epochs, i+1, total_step, loss.item()))

    #def test(self, test_dataset):
    #    self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)
    #    with torch.no_grad():
    #        correct = 0
    #        total = 0
    #        for images, labels in test_loader:
    #            # load data
    #            labels = labels.to(device)
    #            outputs = model(images)
    #            _, predicted = torch.max(outputs.data, 1)
    #            total += labels.size(0)
    #            correct += (predicted == labels).sum().item()
    #        print('Test Accuracy : {} %'.format(100 * correct / total))

    def save(self,):
        torch.save(self.model.state_dict(), 'model.ckpt')

