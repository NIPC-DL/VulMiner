#!/usr/bin/env python3
## -*- coding: utf-8 -*-
import utils
import torch
import numpy as np
from configer import configer
from logger import logger
from torch.utils.data import Dataset, DataLoader


class VulDataset(Dataset):
    def __init__(self, dataset):
        #dataset = np.load('../Cache/dataset.npz')
        #x_set, y_set = dataset['arr_0'], dataset['arr_1']
        #rng = np.arange(x_set.shape[0])
        #np.random.shuffle(rng)
        #x_set_r = []
        #y_set_r = []
        #for i in rng:
        #    x_set_r.append(x_set[i])
        #    y_set_r.append(y_set[i])
        #x_set_r = np.asarray(x_set_r)
        #y_set_r = np.asarray(y_set_r)
        #per = round(len(x_set_r) * 0.8)
        self.X = torch.tensor(dataset[0]).float()
        self.Y = torch.tensor(dataset[1]).long()

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)

