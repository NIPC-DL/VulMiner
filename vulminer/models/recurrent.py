# -*- coding: utf-8 -*-
"""
recurrent.py - Models of Recurrent Neural Network

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""
import torch
import torch.nn as nn

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
            **kwargs):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, **kwargs)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Forward propagate LSTM
        out, _ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 **kwargs):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers,
                          **kwargs)
        # self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense3 = nn.Linear(self.hidden_size, self.num_classes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax()


    def forward(self, x):
        # out = nn.utils.rnn.pack_padded_sequence(x, self.input_size, batch_first=True)
        out, _ = self.gru(x)
        out = self.dense1(out[:, -1, :])
        out = self.relu1(out)
        out = self.dense2(out)
        out = self.relu2(out)
        out = self.dense3(out)
        out = self.softmax(out)
        return out


class BGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 **kwargs):
        super(BGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bgru = nn.GRU(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            bidirectional=True,
            **kwargs)
        self.dense1 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dense2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense3 = nn.Linear(self.hidden_size, self.num_classes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax()


    def forward(self, input, lengths):
        out = torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
        out, _ = self.bgru(out)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = self.dense1(out[:, -1, :])
        out = self.relu1(out)
        out = self.dense2(out)
        out = self.relu2(out)
        out = self.dense3(out)
        out = self.softmax(out)
        return out


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 **kwargs):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            **kwargs,
        )
        self.dense = nn.Linear(
            self.hidden_size,
            self.num_classes,
        )

    def forward(self, input):
        output, _ = self.lstm(input)
        output = self.dense(output[:, -1, :])
        return output


class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 **kwargs):
        super(BLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.blstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            bidirectional=True,
            **kwargs)
        self.dense = nn.Linear(
            self.hidden_size * 2,
            self.num_classes,
        )

    def forward(self, input):
        output, _ = self.blstm(input)
        output = self.dense(output[:, -1, :])
        return output
