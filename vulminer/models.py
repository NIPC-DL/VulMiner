#!/usr/bin/env python3
#coding: utf-8

import torch
import torch.nn as nn
from configer import configer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, batch_size):
        super(BGRU, self).__init__()
        # config
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.batch_size = batch_size
        # layer
        self.bgru = nn.GRU(
                self.input_size,
                self.hidden_size,
                self.num_layers,
                batch_first=True,
                dropout=0.2,
                bidirectional=True
                )
        self.dense = nn.Linear(
                self.hidden_size*2,
                self.num_classes
                )
        self.hidden = self.init_hidden()

    def forward(self, train_dataset):
        output, self.hidden = self.bgru(train_dataset, self.hidden)
        output = self.dense(output)
        #output = output[:,-1,:]
        return output, hidden

    def init_hidden(self):
        hidden = torch.autograd.Variable(torch.zeros(self.num_layers*2, self.batch_size, self.hidden_size))
        return hidden

