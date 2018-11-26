# -*- coding: utf-8 -*-
"""
trainer.py - 

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""
import os
import torch
import numpy as np
import pathlib
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss
from vulminer.utils import logger
from .treeloader import TreeLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    """Manage all things about model training
    
    Args:
        root (str): The work path for trainer

    """

    def __init__(self, root):
        self._root = pathlib.Path(root).expanduser()
        self._root.mkdir(parents=True, exist_ok=True)
        self._models = []
        self._metrics = {}
        self._events = {
            Events.STARTED: [],
            Events.COMPLETED: [],
            Events.EPOCH_STARTED: [],
            Events.EPOCH_COMPLETED: [],
            Events.EXCEPTION_RAISED: [],
            Events.ITERATION_STARTED: [],
            Events.ITERATION_COMPLETED: []
        }
        self._loader_args = {}

    def addData(self, dataset, valid=None, **kwargs):
        """Add datasets for trainer, must a torch Dataset
        
        Args:
            dataset (torch.utils.data.Dataset): The datsets
            valid (torch.utils.data.Dataset, optional): The valid datasets
            kwargs (dict): The parameter for torch DataLoader
            
        """
        if isinstance(dataset, Dataset):
            self._dataset = dataset
            self._datalen = len(dataset)
        if isinstance(valid, Dataset):
            self._valid = valid
        self._loader_args.update(kwargs)

    def addMetrics(self, metric):
        """Add metrics for trainer
        
        Args:
            metrics (dict): The dict of metrics

        """
        if isinstance(metric, dict):
            self._metrics.update(metric)
        else:
            raise ValueError(f'{metric} is not a dict')

    def addModel(self, model):
        """Add models
        
        Args:
            model (list, dict): A model is a dict contained necessary values
                you can add it one by one or just add a list of models
            
        """
        if isinstance(model, list):
            self._models.extend(model)
        else:
            self._models.append(model)

    def fit(self, folds=None, save=True):
        """Start training
        
        Args:
            folds (int, optional): If set, start n-folds validation
            save (bool, optional): Default True, means save the trained model
 
        """
        saved_path = self._root / 'ModelSave'
        saved_path.mkdir(parents=True, exist_ok=True)
        # create total bar to show total progress
        self._total_bar = tqdm(
            desc='Total',
            total=len(self._models),
            unit='model',
        )
        # train model from models
        for model in self._models:
            model_name = model['nn'].__class__.__name__
            # create model bar to show model progress
            epoch = model['epoch']
            model_len = int(
                np.floor(((folds - 1.0) / folds) * self._datalen * folds *
                         epoch)) if folds else self._datalen
            logger.info(f"Start {model_name} Training")
            self._model_bar = tqdm(
                desc=model_name,
                total=model_len,
                unit='sample',
            )
            # n-folds validation
            if folds and isinstance(folds, int):
                for i in range(folds):
                    logger.info(f'Start [{i+1}/{folds}] fold')
                    train, valid = self._get_loader(folds)
                    # create folds bar to show fold progress
                    self._folds_bar = tqdm(
                        desc=f'fold {i+1}/{folds}',
                        total=int(
                            np.floor(epoch * self._datalen * (
                                (folds - 1) / folds))),
                        unit='samples',
                    )
                    # strat training
                    nn = self._training(model, train, valid=valid, folds=folds)
                    self._folds_bar.close()
                    if save:  # save the trained model
                        torch.save(
                            nn.state_dict(),
                            str(saved_path /
                                f'{model_name.lower()}_nn{i+1}.pt'))
            else:  #just training dataset and valid model py valid set
                self._folds_bar = None
                if 'tree' in model_name.lower():
                    train = TreeLoader(self._dataset, **self._loader_args)
                    valid = TreeLoader(
                        self._valid, **
                        self._loader_args) if self._valid else None
                else:
                    train = DataLoader(self._dataset, **self._loader_args)
                    valid = DataLoader(
                        self._valid, **
                        self._loader_args) if self._valid else None
                nn = self._training(model, train, valid=valid)
                if save:  # save the trained model
                    torch.save(nn.state_dict(),
                               str(saved_path / f'{model_name.lower()}_nn.pt'))
            self._model_bar.close()
            self._total_bar.update(1)
        self._total_bar.close()

    def addEvent(self, event_name, event_handler):
        """Add event handler for trainer
        
        Args:
            event_name (ignite.engine.Events): The event name
            event_handler (callable): The function of event handler
        
        """
        self._events[event_name].append(event_handler)

    def _training(self, model, train, valid=None, folds=None):
        """Training given model
        
        Args:
            model (dict): A dict contained necessary values
            train (torch.utils.data.Dataloader): The train dataloader
            valid (torch.utils.data.Dataloader): The valid dataloader
            folds (int): n-folds validation
            
        """

        nn = model['nn']
        model_name = nn.__class__.__name__
        opt = model['optimizer']
        l = model['loss']
        epoch = model['epoch']
        # add loss as basic metric
        self.addMetrics({'loss': Loss(l)})
        # create trainer and evaluator
        trainer = create_supervised_trainer(nn, opt, l, device=device)
        evaluator = create_supervised_evaluator(
            nn, metrics=self._metrics, device=device)
        # add event handler to trainer
        for event_name, event_handler in self._events.items():
            for event in event_handler:
                trainer.add_event_handler(event_name, event, evaluator, train,
                                          valid)
        # defined default event handler for log and save results
        @trainer.on(Events.ITERATION_COMPLETED)
        def iter_comp(trainer):
            batch_size = train.batch_size
            iter_num = trainer.state.iteration
            self._model_bar.update(batch_size)
            if self._folds_bar:
                self._folds_bar.update(batch_size)
            if self._epoch_bar:
                self._epoch_bar.update(batch_size)
            if iter_num % 10 == 0:
                print(
                    f'\rCurrent Loss {trainer.state.output:.3}'.ljust(50),
                    end="")

        @trainer.on(Events.EPOCH_STARTED)
        def epoch_start(trainer):
            self._epoch_bar = tqdm(
                desc=f'epoch {trainer.state.epoch}',
                total=int(np.floor(self._datalen * ((folds - 1) / folds))),
                unit='samples',
            )

        @trainer.on(Events.EPOCH_COMPLETED)
        def train_result(trainer):
            evaluator.run(train)
            metrics = evaluator.state.metrics
            res = []
            for k, v in metrics.items():
                res.append(f'{k}: {v:.3}')
            met_str = ', '.join(res)
            logger.info(
                f"Train Result - Epoch: {trainer.state.epoch}\n{met_str}")
            if self._epoch_bar:
                self._epoch_bar.close()
            print(f"Train Result - Epoch: {trainer.state.epoch}\n{met_str}".
                  ljust(50))

        @trainer.on(Events.EPOCH_COMPLETED)
        def valid_result(trainer):
            if valid:
                evaluator.run(valid)
                metrics = evaluator.state.metrics
                res = []
                for k, v in metrics.items():
                    res.append(f'{k}: {v:.3}')
                met_str = ', '.join(res)
                if trainer.state.epoch == epoch:
                    logger.info(f"Valid Result\n{met_str}")
                    print(f"Valid Result\n{met_str}".ljust(50))
                    with open(
                            str(self._root / 'result.txt'), 'a',
                            encoding='utf-8') as f:
                        f.write(f"{model_name}\nValid Result\n{met_str}\n")

        # start training
        trainer.run(train, max_epochs=epoch)
        return nn

    def _get_loader(self, folds):
        """Get dataloader by given folds
        
        Args:
            folds (int): n-folds validation
        
        """
        size = len(self._dataset)
        indices = list(range(size))
        np.random.shuffle(indices)
        split = int(np.floor(size / folds))
        train_idx, valid_idx = indices[:-split], indices[-split:]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        train_loader = DataLoader(
            self._dataset,
            sampler=train_sampler,
            shuffle=False,
            **self._loader_args)
        valid_loader = DataLoader(
            self._dataset,
            sampler=valid_sampler,
            shuffle=False,
            **self._loader_args)
        return train_loader, valid_loader