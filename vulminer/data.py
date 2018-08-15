#!/usr/bin/env python3
#coding: utf-8

"""
provide data object
"""
import os
import sys
import utils
import hashlib
import tracemalloc
import numpy as np
from logger import logger
from symbolize import cgd2sym
from vectorize import sym2vec


def _cgd_loader(file_list):
    raw_set = []
    cgd_set = []
    for file in file_list:
        logger.debug('Read File: ' + file)
        b = []
        with open(file, 'r') as f:
            for line in f:
                if line != '---------------------------------\n':
                    b.append(line[:-1])
                else:
                    raw_set.append(b)
                    b = []
    for b in raw_set:
        cgd = {}
        cgd['name'] = b[0]
        cgd['codes'] = b[1:-1]
        if len(b[-1]) > 1:
            logger.warn('label data worry: ' + b[0])
        cgd['label'] = b[-1]
        cgd_set.append(cgd)
    logger.info(str(len(cgd_set)) + 'cgd found')
    return cgd_set[:5000]

def _get_file_hash(file_path):
    m = hashlib.md5()
    with open(file_path, 'rb') as f:
        while True:
            data =f.read(2048)
            if not data:
                break
            m.update(data)
    return str(m.hexdigest())

class Data:
    def __init__(self):
        self._cgd_set = None
        self._sym_set = None
        self._vec_set = None
        self.x_set = None
        self.y_set = None

    def load(self, path, type_):
        true_path = os.path.expanduser(path) # expand ~
        file_list = []
        hash_list = []
        # get absolute path and hash
        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.abspath(os.path.join(root, file))
                file_list.append(file_path)
                hash_list.append(_get_file_hash(file_path))
        # get total hash
        m = hashlib.md5()
        total_hash = ''.join(hash_list)
        m.update(total_hash.encode('utf-8'))
        total_hash = str(m.hexdigest())
        logger.debug('total_hash: ' + total_hash)
        record_hash = None
        with open('.cache/data.record', 'a+') as f:
            f.seek(0)
            record_hash = f.read()
            f.seek(0)
            f.truncate()
            f.seek(0)
            f.write(total_hash)
            logger.debug('record_hash: ' + record_hash)
        if record_hash == total_hash:
            logger.info('load data cache')
            self._load_cache()
        else:
            logger.info('load file')
            self._load_file(file_list, type_)

    def _load_cache(self):
        set_ = np.load('.cache/dataset.npz')
        self.x_set = set_['arr_0']
        self.y_set = set_['arr_1']

    def _load_file(self, file_list, type_):
        if type_ == 'sc':
            logger.info('<source code> file load')
            pass
        elif type_ == 'cgd':
            logger.info('<code gadget> file load')
            self._cgd_set = _cgd_loader(file_list)
            self._prep()
        else:
            logger.error('worry file type')

    def _prep(self):
        self._sym_set = cgd2sym(self._cgd_set)
        self._vec_set = sym2vec(self._sym_set)
        x = []
        y = []
        for k, i in enumerate(self._vec_set):
            x.append(i['vector'])
            tmp = i['label']
            y.append(int(tmp, 2))
        self.x_set = np.array(x)
        self.y_set = np.array(y)
        # save set into cache
        np.savez('.cache/dataset.npz', self.x_set, self.y_set)

