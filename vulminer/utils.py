#!/usr/bin/env python3
#coding: utf-8

"""
Utils Function
"""

import re
import yaml
import pprint
import torch
import hashlib


def line_split(line):
    return list(filter(lambda x: x not in [None, '', ' ', ';', '*'], re.split(r'(\(|\)|\s|\,|\;|\[|\*)', line)))

def line_split_plus(line):
    return list(filter(lambda x: x not in [None, '', ' '], re.split(r'(\*|\;|\!|\&|\,|\s|\(|\]|\[|\)|\-\>)', line)))

def remove_symbol(word):
    return re.sub('[\s!&-,;\)\*\[\]\(]', '', word)

def ppt(s):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(s)

def file_md5(file):
    m = hashlib.md5()
    with open(file, 'rb') as f:
        while True:
            data =f.read(2048)
            if not data:
                break
            m.update(data)
    return str(m.hexdigest())

def yaml_load(path):
    try:
        with open(path, 'r') as f:
            data = yaml.load(f)
    except FileNotFoundError:
        data = None
    return data

def yaml_dump(data, path):
    try:
        with open(path, 'w') as f:
            yaml.dump(data, f)
    except FileNotFoundError:
        logger.error('no path found')

def one_hot_embedding(labels, num_classes):
    hot = torch.eye(num_classes)
    return list(hot[int(labels)])
