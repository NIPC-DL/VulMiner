#!/usr/bin/env python3
#coding: utf-8

import click
import utils
from data import Data

"""
Entry Point
"""

@click.command()
@click.argument('path')
@click.option('-t', default = 'sc', help = 'file type')
def main(path, t):
    #vm = Miner()
    #vm.load(path, t)
    #vm.train()
    dataset = Data()
    dataset.load(path, t)
    dataset.prep()
    print(dataset._vec_set[0]['vector'])
    print(dataset._vec_set[0]['vector'][0])


if __name__ == '__main__':
    main()
