# -*- coding: utf-8 -*-
"""
trainer.py - description

Author: Verf
Email: verf@protonmail.com
License: MIT
"""
import torch
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
