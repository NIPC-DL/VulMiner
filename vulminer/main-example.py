# -*- coding: utf-8 -*-
"""
Main entry of VulMiner

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""
import os
from covec.datasets import SySeVR
from covec.processor import TextModel, Word2Vec
from torch.utils.data import DataLoader
from vulminer.utils import configer, logger


def main(argv):
    configer.load(argv[1])
    embedder = Word2Vec(size=20, min_count=1, workers=12)
    processor = TextModel(embedder)
    dataset = SySeVR('~/WorkSpace/Test/', processor, category=['AF'])
    train, valid = dataset.torchset(10)
    train_loader = DataLoader(train, batch_size=50)
    valid_loader = DataLoader(valid, batch_size=50)
