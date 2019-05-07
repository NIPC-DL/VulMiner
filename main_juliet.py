# -*- coding: utf-8 -*-
import pathlib
import torch
import torch.nn as nn
import torch.utils.data as data
import torchplp.utils.utils as utils
import sklearn.metrics as skm
from torchplp.datasets import Juliet, SySeVR
from torchplp.processor.treemodel import TreeModel
from torchplp.processor.textmodel import TextModel
from torchplp.models import jsix
import torchplp.processor.functional as F
from vulminer.utils import logger
from vulminer.trainer import Trainer, predictor
from vulminer.models.recurrent import BGRU

data_path = pathlib.Path('~/Workspace/Test/Data').expanduser()
root_path = pathlib.Path('~/Workspace/Test/').expanduser()

input_size = 100
hidden_size = 200
num_layers = 2
num_classes = 2
learning_rate = 0.0001
dropout = 0.5
batch_size = 50
epochs = 50

category = ['CWE121']

metrics = {
        'acc': skm.accuracy_score,
        'pre': skm.precision_score,
        'f1': skm.f1_score,
        }

bgru = BGRU(input_size, hidden_size, num_layers, num_classes, dropout)
models = [
    {
        'nn': bgru,
        'opti': torch.optim.Adam(bgru.parameters(), lr=learning_rate),
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
            l = list()
            for i in self._x:
                l.append(len(i) if len(i) < length else length)
            self._l = torch.Tensor(l).long()
            self._length = length
        else:
            l = [len(i) for i in self._x]
            self._l = torch.Tensor(l).long()
            self._length = max(self._l)

    def __getitem__(self, index):
        x = self._x[index]
        l = self._l[index]
        y = self._y[index]
        x = utils.truncate_and_padding(x, self._length, 100)
        x = torch.Tensor(x).float()
        return x, l, y

    def __len__(self):
        return len(self._x)


def main():
    logger.level = 'debug'
    logger.addCmdHandler()
    logger.addFileHandler(path=str(root_path / 'vulminer.log'))

    juliet = Juliet(data_path)
    pr = TreeModel()
    for c, samps in juliet.load(category):
        samples, labels = zip(*samps)
        samples = [F.standardize(x) for x in samples]
        samples = [F.tree2seq(x) for x in samples]
        with open(f'{c}.txt', 'w') as f:
            for sample in samples:
                for node in sample:
                    f.write(f'({node.data, node.kind}),')
                f.write('\n')
    print('done')

    # train, valid, tests = utils.spliter(juliet.load(category), ratio=[6,1,1])
    # train_set = MySet(train, processor=pr)
    # valid_set = MySet(valid, processor=pr)
    # tests_set = MySet(tests, processor=pr)
    # train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # valid_loader = data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    # tests_loader = data.DataLoader(tests_set, batch_size=batch_size, shuffle=False)

    # trainer = Trainer(root_path)
    # trainer.addData(train_loader, valid_loader)
    # trainer.addModel(models)
    # trainer.addMetrics(metrics)
    # mod = trainer.fit()

    # predictor(mod, tests_loader, metrics=metrics)
    
if __name__ == '__main__':
    main()
