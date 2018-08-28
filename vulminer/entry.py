#!/usr/bin/env python3
#coding: utf-8

import click
import utils
import numpy as np
from logger import logger
from manager import DataManager
from configer import configer
from dataset import VulDataset
from trainer import Trainer

"""
Entry Point, using click module to create command line app
"""

BANNER = """
         _   _ _____ _____   _____      _____  _
        | \ | |_   _|  __ \ / ____|    |  __ \| |
        |  \| | | | | |__) | |   ______| |  | | |
        | . ` | | | |  ___/| |  |______| |  | | |
        | |\  |_| |_| |    | |____     | |__| | |____
        |_| \_|_____|_|     \_____|    |_____/|______|
"""

@click.command()
@click.option('-c', help = 'config file path')
@click.option('-l', default = '.log', help = 'log file path')
def main(c, l):
    print(BANNER)
    if c:
        configer.init(c)
    else:
        logger.error('no config found')

    #data_manager = DataManager()
    dataset = VulDataset()

    trainer = Trainer()
    trainer.init()
    trainer.fit(dataset)

if __name__ == '__main__':
    main()
