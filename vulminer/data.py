#!/usr/bin/env python3
#coding: utf-8

"""
provide data object
"""
import os
import utils
from symbolize import cgd2sym
from vectorize import sym2vec


def _cgd_loader(file_list):
    raw_set = []
    cgd_set = []
    for file in file_list:
        b = []
        with open(file, 'r') as f:
            for line in f:
                if line != '\n':
                    b.append(line[:-1])
                else:
                    raw_set.append(b)
                    b = []
    for b in raw_set:
        cgd = {}
        cgd['name'] = b[0]
        cgd['codes'] = b[1:-1]
        cgd['label'] = b[-1]
        cgd_set.append(cgd)
    return cgd_set

def _get_abs_file_path(path):
    fl = []
    for root, _, files in os.walk(path):
        for f in files:
            fl.append(os.path.abspath(os.path.join(root, f)))
    fl.sort()
    return fl

class Data:
    def __init__(self):
        self._cgd_set = None
        self._sym_set = None
        self._vec_set = None

    def load(self, path, type_):
        true_path = os.path.expanduser(path)
        file_list = _get_abs_file_path(true_path)
        if type_ == 'sc':
            pass
        elif type_ == 'cgd':
            self._cgd_set = _cgd_loader(file_list)
        else:
            print("Worry Type")

    def prep(self):
        self._sym_set = cgd2sym(self._cgd_set)
        self._vec_set = sym2vec(self._sym_set)

