#!/usr/bin/env python3
#coding: utf-8

def ps_loader(file):
    with open(file, 'r') as f:
        for line in f:
            pass

def preprocessor(file, type_):
    if type_ == 'ps':
        ps_set = ps_loader(file)
