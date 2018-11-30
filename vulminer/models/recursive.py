# -*- coding: utf-8 -*-
"""
recursive.py - Models of Recursive Neural Network

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""
import torch
import torch.nn as nn


class CSTreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CSTreeLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.tanh = nn.Tanh()
        self.sigmod = nn.Sigmoid()
        self.wi = nn.Linear(input_size + hidden_size, hidden_size)
        self.wf = nn.Linear(input_size + hidden_size, hidden_size)
        self.wo = nn.Linear(input_size + hidden_size, hidden_size)
        self.wu = nn.Linear(input_size + hidden_size, hidden_size)

    def node_forward(self, input):
        ch = [child.h for child in input.children]
        cc = [child.c for child in input.children]
        h = sum(ch)
        x = input.vector
        xh = torch.cat((x, h), 0)
        i = self.sigmod(self.wi(xh))
        fl = [self.sigmod(self.wf(torch.cat((x, k), 0))) for k in ch]
        o = self.sigmod(self.wo(xh))
        u = self.tanh(self.wu(xh))
        c = i * u + sum([f * c for f, c in zip(fl, cc)])
        h = o * self.tanh(c)
        return h, c

    def leaf_forward(self, input):
        h = torch.zeros(self.hidden_size)
        x = input.vector
        xh = torch.cat((x, h), 0)
        i = self.sigmod(self.wi(xh))
        o = self.sigmod(self.wo(xh))
        u = self.tanh(self.wu(xh))
        c = i * u
        h = o * self.tanh(c)
        return h, c

    def forward(self, input):
        for child in input.children:
            if child.children:
                child.h, child.c = self.forward(child)
            else:
                child.h, child.c = self.leaf_forward(child)
        return self.node_forward(input)


class TNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classses):
        super(TNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classses = num_classses

        self.tlstm = CSTreeLSTM(input_size, hidden_size)
        self.dense = nn.Linear(hidden_size, num_classses)

    def forward(self, input):
        output, _ = self.tlstm(input)
        output = self.dense(output)
        return output