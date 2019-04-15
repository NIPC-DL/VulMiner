#coding: utf-8
import pathlib
from multiprocessing import cpu_count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import sklearn.metrics as skm
from torchplp.datasets import VulDeePecker
from torchplp.processor.textmodel import TextModel
from torchplp.processor.embedder import Word2Vec
from vulminer.trainer import Trainer
from vulminer.models.recurrent import BLSTM
from vulminer.utils.logger import logger
from vulminer.utils.metrics import fnr, fpr

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_root = pathlib.Path('~/Workspace/Test2/Data').expanduser()
vulm_root = pathlib.Path('~/Workspace/Test2/').expanduser()

input_size = 100
hidden_size = 200
num_layers = 2
num_classes = 2
dropout = 0.5
learning_rate = 0.001
batch_size = 10
epochs = 50
coe = [6,1,1]

word2vec_para = {
    'size': input_size,
    'min_count': 1,
    'workers': cpu_count(),
    }

blstm = BLSTM(input_size, hidden_size, num_layers, num_classes, dropout)

metrics = {
        'accurary': skm.accuracy_score,
        'precision': skm.precision_score,
        'recall': skm.recall_score,
        'fnr': fnr,
        'fpr': fpr,
        'f1': skm.f1_score,
        }

models = [
    {
        'nn': blstm,
        'opti': optim.Adamax(blstm.parameters(), lr=learning_rate),#, momentum=0.9),
        'crit': nn.CrossEntropyLoss(),
        'epochs': epochs,
    },
]

def main():
    logger.addFileHandler(vulm_root / 'vulminer.log')
    logger.addCmdHandler()
    wm = Word2Vec(**word2vec_para)
    processor = TextModel(wm, 50)
    dataset = VulDeePecker(data_root, processor)
    train, valid = dataset.load(folds=5)
    train_loader = data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    trainer = Trainer(str(vulm_root), padded=False)
    trainer.addData(train_loader, valid_loader)
    trainer.addMetrics(metrics)
    trainer.addModel(models)
    mod = trainer.fit()


if __name__ == '__main__':
    main()
