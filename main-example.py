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
from multiprocessing import cpu_count
from covec.datasets import SySeVR, Juliet
from covec.processor import TextModel, TreeModel, Word2Vec
from vulminer.utils import logger
from vulminer.trainer import Trainer
from vulminer.models.recurrent import GRU
from vulminer.models.recursive import CSTLTNN, NTLTNN, CBTNN
from torch.utils.data import DataLoader

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set work path
data_root = pathlib.Path('~/WorkSpace/Test/Vul/Data').expanduser()
vulm_root = pathlib.Path('~/WorkSpace/Test/Vul/').expanduser()

# nn parameter
input_size = 100
hidden_size = 200
num_layers = 1
num_classes = 2
learning_rate = 0.01

# word2vec parameter
word2vec_para = {
    'size': int(input_size / 2),
    'min_count': 1,
    'workers': cpu_count(),
}

# set neural network
cstlstm = CSTLTNN(input_size, hidden_size, num_classes)
ntlstm = NTLTNN(input_size, hidden_size, num_classes)
cbtnn = CBTNN(input_size, num_classes)
gru = GRU(input_size, hidden_size, num_layers, num_classes)

# set models
models = [
    {
        'nn': gru,
        'opti': Adam(gru.parameters(), lr=learning_rate),
        'batch_size': 50,
        'epoch': 1,
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

def main():
    # logger config
    logger.level = 'debug'
    logger.addFileHandler(path=str(vulm_root / 'vulminer.log'))
    # dataset config
    # set word2vec, the parameter same as gensim's word2vec
    embedder = Word2Vec(**word2vec_para)
    # set processor, Text Model need a embedder
    processor = TextModel(embedder)
    # set dataset
    dataset = SySeVR(str(data_root), processor)
    # set trainer
    trainer = Trainer(str(vulm_root))
    # add data and models
    trainer.addData(dataset)
    trainer.addModel(models)
    trainer.fit(['AF'], 10)


if __name__ == '__main__':
    main()
