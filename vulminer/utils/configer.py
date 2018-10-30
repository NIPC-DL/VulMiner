# -*- coding: utf-8 -*-
"""
configer.py - Ths file provide the global configuration for covec

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""
import yaml


class Configer:
    def __init__(self):
        self._config = None

    def __getitem__(self, key):
        return self._config[key]

    def load(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self._config = yaml.load(f)
        except FileNotFoundError:
            print(f'{path} not found')


configer = Configer()