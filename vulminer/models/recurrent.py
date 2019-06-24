# -*- coding: utf-8 -*-
"""
recurrent.py - Models of Recurrent Neural Network

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
            dropout):
        super(BGRU, self).__init__()
        self.bgru = nn.GRU(
                input_size,
                hidden_size,
                num_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=True)
        self.dense = nn.Sequential(
                nn.Linear(hidden_size*2, hidden_size),
                # nn.Tanh(),
                # nn.Linear(hidden_size, num_classes),
                )

    def forward(self, x, l):
        packed_x = pack_padded_sequence(x, l, batch_first=True)
        out, h = self.bgru(packed_x)
        out, idx = pad_packed_sequence(out, batch_first=True)
        idx = (idx-1).view(-1, 1).expand(out.size(0), out.size(2)).unsqueeze(1).long().to(dev)
        out = out.gather(1, idx).squeeze().unsqueeze(1)
        out = self.dense(out[:, -1, :])
        # out = self.dense(h[-1])
        return out

class BGRU1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
            dropout):
        super(BGRU1, self).__init__()
        self.bgru = nn.GRU(
                input_size,
                hidden_size,
                num_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=True)
        self.dense = nn.Sequential(
                # nn.Linear(hidden_size*2, hidden_size),
                # nn.Tanh(),
                nn.Linear(hidden_size, num_classes),
                )

    def forward(self, x, l):
        packed_x = pack_padded_sequence(x, l, batch_first=True)
        out, h = self.bgru(packed_x)
        # out, idx = pad_packed_sequence(out, batch_first=True)
        # idx = (idx-1).view(-1, 1).expand(out.size(0), out.size(2)).unsqueeze(1).long().to(dev)
        # out = out.gather(1, idx).squeeze().unsqueeze(1)
        # out = self.dense(out[:, -1, :])
        out = self.dense(h[-1])
        return out

class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
            dropout):
        super(BLSTM, self).__init__()
        self.blstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=True)
        self.dense = nn.Sequential(
                nn.Linear(hidden_size*2, num_classes)
                )

    def forward(self, x):
        out, _ = self.blstm(x)
        out = self.dense(out[:, -1, :])
        return out
