# -*- coding: utf-8 -*-
"""
Main entry of VulMiner

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""
import os
import pathlib
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from covec.datasets import SySeVR, Juliet, VulDeePecker
from covec.processor import TextModel, TreeModel, Word2Vec, Tree2Seq
from vulminer.utils import logger
from vulminer.trainer import Trainer
from vulminer.models.recurrent import GRU, BGRU
from vulminer.models.recursive import CSTLTNN, NTLTNN, CBTNN
from vulminer.utils.metrics import accurary, precision, f1, tpr, fpr, fnr

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set work path
data_root = pathlib.Path('~/Workspace/Merge/Test/Data').expanduser()
vulm_root = pathlib.Path('~/Workspace/Merge/Test/').expanduser()

# nn parameter
input_size = 50
hidden_size = 200
num_layers = 2
num_classes = 2
learning_rate = 0.001

# word2vec parameter
word2vec_para = {
    'size': input_size,
    'min_count': 1,
    'workers': cpu_count(),
}

# set neural network
gru = GRU(input_size*2, hidden_size, num_layers, num_classes)
bgru = BGRU(input_size*2, hidden_size, num_layers, num_classes)

# set models
models = [
    {
        'nn': bgru,
        'opti': Adam(gru.parameters(), lr=learning_rate),
        'loss': nn.CrossEntropyLoss(),
        'batch_size': 100,
        'epoch': 10,
    },
    # {
    #     'nn': cstlstm.to(dev),
    #     'opti': Adam(cstlstm.parameters(), lr=learning_rate),
    #     'loss': nn.BCEWithLogitsLoss(),
    #     'batch_size': 50,
    #     'epoch': 5,
    # },
    # {
    #     'nn': ntlstm.to(dev),
    #     'opti': Adam(cstlstm.parameters(), lr=learning_rate),
    #     'loss': nn.BCEWithLogitsLoss(),
    #     'epoch': 5,
    # },
    # {
    #     'nn': cbtnn.to(dev),
    #     'opti': Adam(cstlstm.parameters(), lr=learning_rate),
    #     'loss': nn.BCEWithLogitsLoss(),
    #     'epoch': 5,
    # },
]

metrics = {
        'acc': accurary,
        'pre': precision,
        'f1': f1,
        'tpr': tpr,
        'fpr': fpr,
        'fnr': fnr,
        }

def main():
    # logger config
    logger.level = 'debug'
    logger.addCmdHandler()
    logger.addFileHandler(path=str(vulm_root / 'vulminer.log'))
    # dataset config
    # set word2vec, the parameter same as gensim's word2vec
    embedder = Word2Vec(**word2vec_para)
    # set processor, Text Model need a embedder
    processor = Tree2Seq(embedder, 100)
    # set dataset
    dataset = Juliet(str(data_root), processor)
    # set trainer
    trainer = Trainer(str(vulm_root))
    # add data and models
    trainer.addData(dataset)
    trainer.addModel(models)
    trainer.addMetrics(metrics)
    trainer.fit([
        'CWE121',
        'CWE122',
        'CWE123',
        'CWE124',
        ], 10)


if __name__ == '__main__':
    main()
