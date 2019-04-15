#coding: utf-8
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import sklearn.metrics as skm
from torchplp.datasets import Juliet
from torchplp.processor.treemodel import Treemodel
from vulminer.trainer import Trainer, predictor
from vulminer.models.recurrent import BGRU
from vulminer.utils.logger import logger
from vulminer.utils.metrics import fnr, fpr


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_root = pathlib.Path('~/Workspace/Test/Data').expanduser()
vulm_root = pathlib.Path('~/Workspace/Test/').expanduser()

# nn parameter

input_size = 100
hidden_size = 200
num_layers = 4
num_classes = 2
dropout = 0.5
learning_rate = 0.0001
batch_size = 30
epochs = 50
penalty = 0.0005
coe = [3,1,1]

bgru = BGRU(input_size, hidden_size, num_layers, num_classes, dropout)

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
        'nn': bgru,
        'opti': optim.Adam(
            bgru.parameters(),
            lr=learning_rate,
            weight_decay=penalty),
            # momentum=0.9),
        'crit': nn.CrossEntropyLoss(),
        'epochs': epochs,
    },
]

category = [
        'CWE121',
        'CWE122',
        ]

def main():
    logger.addFileHandler(vulm_root / 'vulminer.log')
    logger.addCmdHandler()
    processor = Treemodel()
    dataset = Juliet(data_root, processor)
    train, valid, tests = dataset.load(category, coe=coe)
    train_loader = data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    tests_loader = data.DataLoader(tests, batch_size=batch_size, shuffle=False)
    logger.info(f'train {len(train)} samps')
    logger.info(f'valid {len(valid)} samps')
    logger.info(f'tests {len(tests)} samps')
    trainer = Trainer(str(vulm_root), padded=True)
    trainer.addData(train_loader, valid_loader)
    trainer.addMetrics(metrics)
    trainer.addModel(models)
    mod = trainer.fit()
    predictor(mod, tests_loader, metrics)
    torch.save(mod.state_dict(), str(vulm_root / 'vulm.model'))



if __name__ == '__main__':
    main()
