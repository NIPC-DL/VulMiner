#!/usr/bin/env python3
## -*- coding: utf-8 -*-
import utils
import torch
import numpy as np
from configer import configer
from logger import logger
from torch.utils.data import Dataset, DataLoader


class VulDataset(Dataset):
    def __init__(self, ):
        dataset = np.load('../Cache/dataset.npz')
        x_set, y_set = dataset['arr_0'], dataset['arr_1']
        tmp_y = []
        for i in y_set:
            tmp_y.append(utils.one_hot_embedding(i, 2))
        self.X = x_set
        self.Y = np.asanyarray(tmp_y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)

