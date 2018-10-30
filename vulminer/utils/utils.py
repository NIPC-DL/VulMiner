# -*- coding: utf-8 -*-
"""
utils.py - The collection of some usefully functions

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""
import os


def ensure_path_exist(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)
    if path[-1] != '/':
        path += '/'
    return path