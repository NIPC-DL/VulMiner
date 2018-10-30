# -*- coding: utf-8 -*-
"""
log_event.py - The collection of builtin log event handler

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""


def log_10iter(trainer, evaluator, train, valid):
    iter_num = trainer.state.iteration
    if iter_num % 10 == 0:
        logger.info()


def log_training_results(trainer, evaluator, train, valid):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    logger.info()


def log_valid(trainer, evaluator, train, valid):
    evaluator.run(valid_loader)
    metrics = evaluator.state.metrics
    logger.info()
