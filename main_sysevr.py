# -*- coding: utf-8 -*-
import pathlib
import torch
import torch.nn as nn
import torch.utils.data as data
import torchplp.utils.utils as utils
import sklearn.metrics as skm
from multiprocessing import cpu_count
from torchplp.datasets import SySeVR
from torchplp.processor.textmodel import TextModel, Word2Vec
from vulminer.utils import logger
from vulminer.trainer import Trainer, predictor
from vulminer.models.recurrent import BGRU, BLSTM
import vulminer.utils.metrics as mm

data_path = pathlib.Path('~/Workspace/Test/Data').expanduser()
root_path = pathlib.Path('~/Workspace/Test/').expanduser()

input_size = 100
hidden_size = 200
num_layers = 2
num_classes = 2
learning_rate = 0.0005
dropout = 0.5
batch_size = 50
epochs = 50

word_size = 100

category = ['FC']

metrics = {
        'acc': skm.accuracy_score,
        'pre': skm.precision_score,
        'f1': skm.f1_score,
        'fnr': mm.fnr,
        'fpr': mm.fpr,
        }

blstm = BLSTM(input_size, hidden_size, num_layers, num_classes, dropout)
models = [
    {
        'nn': blstm,
        'opti': torch.optim.Adam(blstm.parameters(), lr=learning_rate),
        'crit': nn.CrossEntropyLoss(),
        'batch_size': batch_size,
        'epochs': epochs,
    },
    ]

class MySet(data.Dataset):
    def __init__(self, dataset, processor=None, length=None):
        self._x = dataset['x']
        if processor:
            self._x = processor(self._x)
        self._y = torch.Tensor(dataset['y']).long()
        if length is not None:
            self._length = length
        else:
            self._length = max([len(i) for i in self._x])

    def __getitem__(self, index):
        x = self._x[index]
        y = self._y[index]
        x = utils.truncate_and_padding(x, self._length, word_size) 
        x = torch.Tensor(x).float()
        return x, y

    def __len__(self):
        return len(self._x)


def main():
    logger.level = 'debug'
    logger.addCmdHandler()
    logger.addFileHandler(path=str(root_path / 'vulminer.log'))

    dataset = SySeVR(data_path)
    embedder = Word2Vec(size=word_size, min_count=0, workers=cpu_count(), sg=1)
    pr = TextModel(embedder)
    print(f'start load')
    train, valid, tests = utils.spliter(dataset.load(category), ratio=[6,1,1])
    print(f'start processor')
    train_set = MySet(train, processor=pr, length=None)
    valid_set = MySet(valid, processor=pr, length=None)
    tests_set = MySet(tests, processor=pr, length=None)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    tests_loader = data.DataLoader(tests_set, batch_size=batch_size, shuffle=False)

    trainer = Trainer(root_path)
    trainer.addData(train_loader, valid_loader)
    trainer.addModel(models)
    trainer.addMetrics(metrics)
    mod = trainer.fit()

    predictor(mod, tests_loader, metrics=metrics)
    
if __name__ == '__main__':
    main()
