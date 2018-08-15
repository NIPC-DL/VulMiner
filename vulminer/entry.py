#!/usr/bin/env python3
#coding: utf-8

import click
import utils
import configparser
import numpy as np
from logger import logger
from data import Data
from training import Trainer

"""
Entry Point, using click module to create command line app
"""

BANNER = [" _   _ _____ _____   _____      _____  _    ",
        "| \ | |_   _|  __ \ / ____|    |  __ \| |     ",
        "|  \| | | | | |__) | |   ______| |  | | |     ",
        "| . ` | | | |  ___/| |  |______| |  | | |     ",
        "| |\  |_| |_| |    | |____     | |__| | |____ ",
        "|_| \_|_____|_|     \_____|    |_____/|______|"]


@click.command()
@click.option('-c', help = 'config file path')
@click.option('-l', default = '.log', help = 'log file path')
def main(c, l):
    for i in BANNER:
        print(i)

    config = configparser.ConfigParser()
    if c:
        config.read(c)
    else:
        logger.error('no config found')

    dataset = Data()
    dataset.load(config['Input']['data_path'], config['Input']['data_type'])

    trainer = Trainer(config)
    trainer.load_data(dataset.x_set, dataset.y_set)
    trainer.training()
    trainer.model.save('vulmodel.h5')


if __name__ == '__main__':
    main()
