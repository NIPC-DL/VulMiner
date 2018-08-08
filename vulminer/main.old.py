#!/usr/bin/env python3
#coding: utf-8

"""
Main Entry
"""

import sys
import re
from loader import Loader
from preper import Preper
from transfer import Transfer
from trainer import Trainer
import numpy as np


class Vulminer:
    def __init__(self):
        """
        The structure for set
        set = [
            name,
            [code blocks],
            result
        ]
        """
        self._cgd_set = None # curpos for code gadget
        self._sym_set = None # curpos for symbolic representations
        self._sentence_curpos = None # curpos for code sentence
        self._vec_model = None # curpos for vector representations

    def load(self, file_path):
        """
        Load file and transform their content to code gadget
        """
        loader = Loader(file_path)
        self._cgd_set = loader.get_cgd()

    def prep(self):
        """
        Code preprocess, transform code gadget to symbolic representations
        """
        preper = Preper(self._cgd_set)
        self._sym_set = preper.get_symr()

    def trans(self):
        """
        word2vec, transform symbolic to vector
        """
        transfer = Transfer(self._sym_set)
        self._sentence_curpos = transfer._sentence_curpos
        self._vec_model = transfer.model
        ll = []
        for i in transfer._sym_split_set:
            l = 0
            for j in i[1]:
                l += len(j)
            ll.append(l)
        data = np.array(ll)
        print("mean: ", np.mean(data))
        print("median: ", np.median(data))
        print("pre: ", np.percentile(data, 95))



    def train(self):
        """
        Training Model
        """
        trainer = Trainer(self._vec_set)
        model = trainer.get_model()

    def test(self):
        """
        Test
        """
        pass

def main():
    pass


if __name__ == '__main__':
    main()
    #vm = Vulminer()
    #vm.load(sys.argv[1])
    #vm.prep()
    #vm.trans()
    #with open('sym_list.txt', 'wt') as f:
    #    for i in vm._sym_set:
    #        f.write(i[0] + '\n')
    #        for j in i[1]:
    #            f.write(j + '\n')
    #        f.write(i[2] + '\n')
    #        f.write('\n')
    #with open('cgd_list.txt', 'wt') as f:
    #    for i in vm._cgd_set:
    #        f.write(i[0] + '\n')
    #        for j in i[1]:
    #            f.write(j + '\n')
    #        f.write(i[2] + '\n')
    #        f.write('\n')
    #vm.train()
    #vm.test()
