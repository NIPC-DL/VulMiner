#!/usr/bin/env python3
#coding: utf-8

import torch
import torch.nn as nn
from configer import configer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, batch_size, dropout):
        super(BGRU, self).__init__()
        # config
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.dropout = dropout
        # layer
        self.bgru = nn.GRU(
                self.input_size,
                self.hidden_size,
                self.num_layers,
                dropout=self.dropout,
                batch_first=True,
                bidirectional=True
                )
        self.dense = nn.Linear(
                self.hidden_size*2,
                self.num_classes,
                )
        #self.softmax = nn.Softmax(dim=0)

    def forward(self, train_dataset):
        hidden = torch.zeros(self.num_layers*2, train_dataset.size(0), self.hidden_size).to(device)
        output, _ = self.bgru(train_dataset, hidden)
        output = self.dense(output[:, -1, :])
        #output = self.softmax(output)
        return output

