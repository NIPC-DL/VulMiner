# -*- coding: utf-8 -*-
"""
recursive.py - Models of Recursive Neural Network

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    def node_forward(self, x, hl, cl):
        h = sum(hl).to(dev)
        xh = torch.cat((x, h), 0).to(dev)
        i = self.sigmod(self.wi(xh))
        fl = [self.sigmod(self.wf(torch.cat((x, k), 0))) for k in cl]
        o = self.sigmod(self.wo(xh))
        u = self.tanh(self.wu(xh))
        c = i * u + sum([f * c for f, c in zip(fl, cl)])
        h = o * self.tanh(c)
        return h, c

    def leaf_forward(self, x):
        h = torch.zeros(self.hidden_size).to(dev)
        xh = torch.cat((x, h), 0).to(dev)
        i = self.sigmod(self.wi(xh))
        o = self.sigmod(self.wo(xh))
        u = self.tanh(self.wu(xh))
        c = i * u
        h = o * self.tanh(c)
        return h, c

    def forward(self, input):
        hl = []
        cl = []
        for child in input.children:
            if child.children:
                h, c = self.forward(child)
            else:
                h, c = self.leaf_forward(child.vector)
            hl.append(h)
            cl.append(c)
        return self.node_forward(input.vector, hl, cl)


class CSTLTNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classses):
        super(CSTLTNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classses = num_classses

        self.tlstm = CSTreeLSTM(input_size, hidden_size)
        self.dense = nn.Linear(hidden_size, num_classses)

    def forward(self, input):
        output, _ = self.tlstm(input)
        output = self.dense(output)
        return output


class NaryTreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NaryTreeLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.tanh = nn.Tanh()
        self.sigmod = nn.Sigmoid()
        self.wi = nn.Linear(input_size, hidden_size)
        self.wf = nn.Linear(input_size, hidden_size)
        self.wo = nn.Linear(input_size, hidden_size)
        self.wu = nn.Linear(input_size, hidden_size)
        self.wf_num = 0
        self.ui = nn.Linear(hidden_size, hidden_size, bias=False)
        self.uo = nn.Linear(hidden_size, hidden_size, bias=False)
        self.uu = nn.Linear(hidden_size, hidden_size, bias=False)

    def node_forward(self, x, hl, cl):
        if len(hl) > self.wf_num:
            for i in range(len(hl) - self.wf_num):
                setattr(
                    self, f'wf{i+self.wf_num+1}',
                    nn.Linear(self.hidden_size, self.hidden_size,
                              bias=False).to(dev))
            self.wf_num = len(hl)
        uih = sum(map(self.ui, hl)).to(dev)
        i = self.sigmod(self.wi(x) + uih)
        fl = []
        for ind, val in enumerate(hl):
            wf = getattr(self, f'wf{ind+1}')
            ukfh = sum(map(wf, hl)).to(dev)
            fk = self.sigmod(self.wf(x) + ukfh)
            fl.append(fk)
        uoh = sum(map(self.uo, hl)).to(dev)
        o = self.sigmod(self.wo(x) + uoh)
        uuh = sum(map(self.uu, hl)).to(dev)
        u = self.tanh(self.wu(x) + uuh)
        c = i * u + sum([f * c for f, c in zip(fl, cl)])
        h = o * self.tanh(c)
        return h, c

    def leaf_forward(self, x):
        i = self.sigmod(self.wi(x))
        o = self.sigmod(self.wo(x))
        u = self.tanh(self.wu(x))
        c = i * u
        h = o * self.tanh(c)
        return h, c

    def forward(self, input):
        hl = []
        cl = []
        for child in input.children:
            if child.children:
                h, c = self.forward(child)
            else:
                h, c = self.leaf_forward(child.vector)
            hl.append(h)
            cl.append(c)
        return self.node_forward(input.vector, hl, cl)


class NTLTNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classses):
        super(NTLTNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classses = num_classses

        self.tlstm = NaryTreeLSTM(input_size, hidden_size)
        self.dense = nn.Linear(hidden_size, num_classses)

    def forward(self, input):
        output, _ = self.tlstm(input)
        output = self.dense(output)
        return output


class CBTree(nn.Module):
    def __init__(self, input_size):
        super(CBTree, self).__init__()
        self.input_size = input_size

        self.tanh = nn.Tanh()
        self.wl = nn.Linear(input_size, input_size, bias=False)
        self.wr = nn.Linear(input_size, input_size, bias=False)

    def node_forward(self, x, hl):
        tot = len(hl)
        ch = []
        for ind, val in enumerate(hl):
            if tot > 1:
                lc, rc = self._coe(ind + 1, tot)
            else:
                lc = rc = 0.5
            ch.append(lc * self.wl(val) + rc * self.wr(val))
        h = x + sum(ch)
        h = self.tanh(h)
        return h

    def forward(self, input):
        hl = list()
        for child in input.children:
            if child.children:
                hl.append(self.forward(child))
            else:
                hl.append(child.vector)
        return self.node_forward(input.vector, hl)

    @staticmethod
    def _coe(ind, tot):
        lc = (tot - ind) / (tot - 1)
        rc = (ind - 1) / (tot - 1)
        return lc, rc


class CBTNN(nn.Module):
    def __init__(self, input_size, num_classses):
        super(CBTNN, self).__init__()
        self.input_size = input_size
        self.num_classses = num_classses

        self.sigmod = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.tnn = CBTree(input_size)
        self.dense = nn.Linear(input_size, num_classses)

    def forward(self, input):
        output = self.tnn(input)
        output = self.dense(output)
        output = self.sigmod(output)
        return output


class NTLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        pass

    def forward(self, input):
        hl = []
        cl = []
        for child in input.children:
            if child.children:
                h, c = self.forward(child)
            else:
                h, c = self.leaf_forward(child.vector)
            hl.append(h)
            cl.append(c)
        return self.node_forward(input.vector, hl, cl)
