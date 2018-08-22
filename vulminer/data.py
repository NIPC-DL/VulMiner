#!/usr/bin/env python3
#coding: utf-8

"""
provide data object
"""
import os
import sys
import utils
import hashlib
import numpy as np
from glob import glob
from logger import logger
#from symbolize import cgd2sym
#from vectorize import sym2vec


class Data(object):
    def __init__(self):
        pass

    def load(self, config):
        self._config = config
        file_record = utils.yaml_load('.cache/record.yaml')
        if not file_record:
            file_record = []
        cache_flag = False
        file_list = self._get_file_list(config)
        for path, __ in file_list:
            fhash = utils.file_md5(path)
            if fhash not in file_record:
                cache_flag = True
                file_record.append(fhash)
                logger.info('New file {0} found, add to cache'.format(path))
        utils.yaml_dump(file_record, '.cache/record.yaml')
        if cache_flag:
            logger.info('Cache data update success')
        else:
            logger.info('Faild to find new data, use cache data')

    def _get_file_list(self):
        """
        get file list from config
        """
        file_list = []
        for path, type_ in self._config:
            if os.path.isdir(path):
                for file in glob(path + '/*.*'):
                    file_list.append([file, type_])
            elif os.path.isfile(path):
                file_path.append([file, type_])
            else:
                logger.warn("worry file path")
        return file_list

    def _file_loader(self, path, type_):
        if type_ == 'ps':
            with open(path, 'r') as f:
                for line in f:
                    pass
