#!/usr/bin/env python3
#coding: utf-8

import click
import utils
import numpy as np
from logger import logger
from data import Data
#from training import Trainer

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
        config = utils.yaml_load(c)
    else:
        logger.error('no config found')

    dataset = Data()
    dataset.load(config['Data'])

    #trainer = Trainer()
    #trainer.init(config)
    #trainer.load(dataset.x_set, dataset.y_set)
    #trainer.fit()

if __name__ == '__main__':
    main()
