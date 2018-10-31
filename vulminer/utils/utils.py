# -*- coding: utf-8 -*-
"""
utils.py - The collection of some usefully functions

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""
import os


def path_check(path):
    """Check and ensure the path exists"""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)
    if path[-1] != '/':
        path += '/'
    return path