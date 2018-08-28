#!/usr/bin/env python3
## -*- coding: utf-8 -*-
import utils
import logger


class Config(object):
    def __init__(self):
        self.Data = None
        self.Model = None

    def init(self, path):
        config = utils.yaml_load(path)
        self.Data = config['Data']
        self.Model = config['Model']

    def getData(self):
        if self.Data:
            return self.Data
        else:
            logger.warn("You need init config first")

    def getModel(self):
        if self.Model:
            return self.Model
        else:
            logger.warn("You need init config first")

configer = Config()

