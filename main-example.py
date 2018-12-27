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
from ignite.engine import Events
from multiprocessing import cpu_count
from covec.datasets import SySeVR, Juliet
from covec.processor import TextModel, TreeModel, Word2Vec
from ignite.metrics import Precision, Recall
from vulminer.utils import logger
from vulminer.trainer import Trainer
from vulminer.models.recursive import CSTLTNN, NTLTNN, CBTNN
from torch.utils.data import DataLoader

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set work path
data_root = pathlib.Path('~/WorkSpace/Test/Vul/Data').expanduser()
vulm_root = pathlib.Path('~/WorkSpace/Test/Vul/').expanduser()

# nn parameter
input_size = 100
hidden_size = 200
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

# set models
models = [
    {
        'nn': cstlstm.to(dev),
        'optimizer': Adam(cstlstm.parameters(), lr=learning_rate),
        'loss': nn.BCEWithLogitsLoss(),
        'batch_size': 50,
        'epoch': 5,
    },
    # {
    #     'nn': ntlstm.to(dev),
    #     'optimizer': Adam(cstlstm.parameters(), lr=learning_rate),
    #     'loss': nn.BCEWithLogitsLoss(),
    #     'epoch': 5,
    # },
    # {
    #     'nn': cbtnn.to(dev),
    #     'optimizer': Adam(cstlstm.parameters(), lr=learning_rate),
    #     'loss': nn.BCEWithLogitsLoss(),
    #     'epoch': 5,
    # },
]

# set metrics
metrics = {
    'prec': Precision(average=True),
    'recall': Recall(average=True),
}


# user-defined event handler
def log_iter_10(trainer, evaluator, train, valid):
    iter_num = trainer.state.iteration
    epoch_num = trainer.state.epoch
    loss = trainer.state.output
    if iter_num % 10 == 0:
        logger.info(f"Epoch[{epoch_num}] Iter: {iter_num} Loss: {loss:.2}")


def main():
    # logger config
    logger.level = 'debug'
    logger.addFileHandler(path=str(vulm_root / 'vulminer.log'))
    # dataset config
    # set word2vec, the parameter same as gensim's word2vec
    embedder = Word2Vec(**word2vec_para)
    # set processor, Text Model need a embedder
    processor = TreeModel(embedder, 100)
    # set dataset ['AE', 'AF', 'AU', 'PU']
    dataset = Juliet(
        str(data_root),
        processor,
        proxy='socks5://127.0.0.1:1080',
        category=['AF'])
    # use processor to process dataset, you can call use
    # get pytorch dataset object
    # trainer config
    trainer = Trainer(str(vulm_root))
    # add dataset, the trainer can only have one dataset
    # the old dataset will replace by the new dataset when called
    # optional, you can input parameter for pytorch dataloader
    trainer.addData(dataset)
    trainer.addModel(models)
    trainer.fit(5)


if __name__ == '__main__':
    main()