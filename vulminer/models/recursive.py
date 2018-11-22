# -*- coding: utf-8 -*-
"""
recursive.py - Models of Recursive Neural Network

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""
import torch
import torch.nn as nn


class TBNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TBNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self._all_weights = []
        for layer in range(num_layers):
            pass


class TreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.tanh = nn.Tanh()
        self.sigmod = nn.Sigmoid()
        self.wi = nn.Linear()
        self.wf = nn.Linear()
        self.wo = nn.Linear()

    def node_forward(self):
        pass