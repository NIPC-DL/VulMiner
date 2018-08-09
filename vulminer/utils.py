#!/usr/bin/env python3
#coding: utf-8

"""
Utils Function
"""

import re
import pprint


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
