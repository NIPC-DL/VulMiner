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
from configer import configer
from preper import prep_sym, prep_wm


class DataManager(object):
    """
    DataManager maintain every data struct and their method.
    """
    def __init__(self):
        self.init()

    def init(self):
        # load record
        if os.path.exists('.cache/record.yaml'):
            record = utils.yaml_load('.cache/record.yaml')
            logger.info('record found, load record')
        else:
            record = {'hash':[],}
            utils.yaml_dump(record, '.cache/record.yaml')
            logger.info('no record found, create new record')
        cache_flag = False
        # get file list
        file_list = self._get_file_list()
        for path, type_ in file_list:
            print(path)
            fhash = utils.file_md5(path)
            assert len(fhash) > 0
            if fhash not in record['hash']:
                # add new data to cache
                cache_flag = True
                record['hash'].append(fhash)
                logger.info('New file {0} {1} found, add to cache'.format(path, type_))
                prep_sym(path, type_)
                logger.info('{0} {1} add to cache success'.format(path, type_))
                utils.yaml_dump(record, '.cache/record.yaml')
        if cache_flag:
            logger.info('Cache data update success')
            prep_wm()
            prep_vec()
        else:
            logger.info('Data up to date, use cache data')

    def load_data(self, num):
        pass


    def _get_file_list(self):
        """
        get file list from config
        """
        file_list = []
        data_config = configer.getData()
        for p, t in data_config:
            if os.path.isdir(p):
                for file in glob(p + '/*.txt'):
                    file_list.append([file, t])
            elif os.path.isfile(p):
                file_list.append([p, t])
            else:
                logger.warn("worry file path")
        print(file_list)
        assert len(file_list) > 0
        return file_list

