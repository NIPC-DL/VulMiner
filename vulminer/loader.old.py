#!/usr/bin/env python3
#coding: utf-8

"""
This file provide the method for file load, and transform them into code gadget
"""
import re
import os
import sys


class Loader:
    def __init__(self, file_path):
        self._file_path = file_path
        self._file_list = os.listdir(file_path)
        self._file_list.sort()
        self._cgd_set = []
        self._txt2list()

    def _txt2list(self):
        """
        Read cgd txt file and prase it into cgd block list
        eg:
            cgd_block[0]: CVE........
            cgd_block[1]: code block
            cgd_block[2]: 1 or 0 as result
        """
        cgd_list = []
        for file_name in self._file_list:
            with open(self._file_path + '/' + file_name, 'rt') as file:
                cgd_block = []
                for line in file:
                    if line != '\n':
                        cgd_block.append(line[:-1])
                    else:
                        cgd_list.append(cgd_block)
                        cgd_block = []
        for b in cgd_list:
            tmp = []
            tmp.append(b[0])
            tmp.append(b[1:-1])
            tmp.append(b[-1])
            self._cgd_set.append(tmp)

    def get_cgd(self):
        return self._cgd_set

