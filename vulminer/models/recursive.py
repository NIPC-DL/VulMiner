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
from collections import deque
from torch.autograd import Variable


class TNNBase(nn.Module):
    def __init__(self, input_size):
        super(TNNBase, self).__init__()
        self.isize = input_size
        self.tanh = nn.Tanh()

    def forward(self, input):
        child_num = len(input.children)
        self.liner_creater(child_num)
        cres = []
        for ind, child in enumerate(input.children):
            cres.append(self.liner[ind](child.data))
        cdata = sum(cres)
        out = self.tanh(cdata + input.data)
        return out

    def liner_creater(self, size):
        self.liner = [
            nn.Linear(self.isize, self.isize, bias=False) for _ in range(size)
        ]


class TNN(nn.Module):
    def __init__(self, input_size, **kwargs):
        super(TNN, self).__init__()
        self.isize = input_size
        self.tnncells = []
        self.cw = nn.Linear(self.isize, self.isize, bias=False)
        self.tanh = nn.Tanh()

    def node_forward(self, input):
        output = sum([self.cw(x.output) for x in input.children])
        output = self.tanh(output + input.vector)
        return output

    def forward(self, input):
        for c in input.children:
            c.output = self.forward(c) if bool(c.children) else c.vector
        return self.node_forward(input)


def treeh(input):
    que = deque()
    que.append(input)
    chs = []
    chs.append(input)
    hight = 1
    while que:
        node = que.popleft()
        flag = False
        for child in node.children:
            if len(child.children) > 0:
                que.append(child)
                chs.append(child)
                flag = True
        if flag:
            hight += 1
        last = node


class Node:
    def __init__(self, i):
        self.parent = None
        self.children = []
        self.data = i


class Ast:
    def __init__(self, data):
        self.parent = None
        self.children = []
        self.data = data
        self.vector = torch.randn(5)
        self.output = None


if __name__ == "__main__":
    # root = Node(0)
    # root.data
    # c1 = Node(1)
    # c2 = Node(2)
    # c3 = Node(3)
    # c4 = Node(4)
    # c5 = Node(5)
    # c6 = Node(6)
    # c7 = Node(7)
    # c8 = Node(8)
    # c9 = Node(9)
    # c7.children = [c8, c9]
    # c8.parent = c7
    # c9.parent = c7
    # c1.children = [c4, c5]
    # c4.parent = c1
    # c5.parent = c1
    # c3.children = [c6, c7]
    # c6.parent = c3
    # c7.parent = c3
    # root.children = [c1, c2, c3]
    # c1.parent = root
    # c2.parent = root
    # c3.parent = root
    # treeh(root)
    r = Ast(0)
    c1 = Ast(1)
    c2 = Ast(2)
    c3 = Ast(3)
    c4 = Ast(4)
    c2.children = [c3, c4]
    c3.parent = c2
    c4.parent = c2
    r.children = [c1, c2]
    c1.parent = r
    c2.parent = r
    m = TNN(5)
    out = m(r)
    print(out)