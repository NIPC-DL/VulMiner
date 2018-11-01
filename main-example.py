# -*- coding: utf-8 -*-
"""
Main entry of VulMiner

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""
import os
import torch.nn as nn
from torch.optim import Adam
from ignite.engine import Events
from multiprocessing import cpu_count
from covec.datasets import SySeVR
from covec.processor import TextModel, Word2Vec
from ignite.metrics import Precision, Recall
from vulminer.utils import logger
from vulminer.trainer import Trainer
from vulminer.models.recurrent import GRU, BGRU

# set work path
data_root = os.path.expanduser('~/WorkSpace/Data')
vulm_root = os.path.expanduser('~/WorkSpace/Test/vulminer/')

# nn parameter
input_size = 50
hidden_size = 50
num_layers = 2
num_classes = 2
dropout = 0.2
learning_rate = 0.1

# word2vec parameter
word2vec_para = {
    'size': input_size,
    'min_count': 1,
    'workers': cpu_count(),
}

# dataloader parameter
dataloader_para = {
    'batch_size': 50,
}

# set neural network
gru = GRU(
    input_size,
    hidden_size,
    num_layers,
    num_classes,
    dropout=dropout,
    batch_first=True,
)
bgru = BGRU(
    input_size,
    hidden_size,
    num_layers,
    num_classes,
    dropout=dropout,
    batch_first=True,
)

# set models
models = [
    {
        'nn': gru,
        'optimizer': Adam(gru.parameters(), lr=learning_rate),
        'loss': nn.CrossEntropyLoss(),
        'epoch': 5,
    },
    {
        'nn': bgru,
        'optimizer': Adam(bgru.parameters(), lr=learning_rate),
        'loss': nn.CrossEntropyLoss(),
        'epoch': 5,
    },
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
    logger.addFileHandler(path=vulm_root + 'vulminer.log')  # set log file path
    # dataset config
    # set word2vec, the parameter same as gensim's word2vec
    embedder = Word2Vec(**word2vec_para)
    # set processor, Text Model need a embedder
    processor = TextModel(embedder)
    # set dataset
    dataset = SySeVR(data_root)
    # use processor to process dataset, you can call use
    dataset.process(processor, category=['AF'])
    # get pytorch dataset object
    torchset = dataset.torchset()
    # trainer config
    trainer = Trainer(vulm_root)
    # add dataset, the trainer can only have one dataset
    # the old dataset will replace by the new dataset when called
    # optional, you can input parameter for pytorch dataloader
    trainer.addData(torchset, **dataloader_para)
    # add metrics
    trainer.addMetrics(metrics)
    # add models
    trainer.addModel(models)
    # add extra event handler for trainer
    trainer.addEvent(Events.ITERATION_COMPLETED, log_iter_10)
    # start training, you can input a int number to start a n-folds validation
    trainer.fit(5)


if __name__ == '__main__':
    main()