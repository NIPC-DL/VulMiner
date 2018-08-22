#!/usr/bin/env python3
#coding: utf-8

"""
Utils Function
"""

import re
import yaml
import pprint
import hashlib


def code_split(code):
    return list(filter(lambda x: x not in [None, ''], re.split(r'(;|,|\s|\(|\[|-\>)', code)))

def line_split(sentence):
    return list(filter(lambda x: x not in [None, ''], re.split(r'(\*|;|!|&|,|\s|\(|\[|-\>)', sentence)))

def remove_blank_and_empty(l):
    return list(filter(lambda x: x not in [None, '', ' '], l))

def remove_symbol(word):
    return re.sub('[!&-,;\)\*\[\]]', '', word)

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
    with open(path, 'r') as f:
        data = yaml.load(f)
    return data

def yaml_dump(data, path):
    with open(path, 'w') as f:
        yaml.dump(data, f)
    return True
