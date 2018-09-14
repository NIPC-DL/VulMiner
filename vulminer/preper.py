#!/usr/bin/env python3
#coding: utf-8

import re
import os
import utils
import gensim
import numpy as np
from logger import logger
from configer import configer
from multiprocessing import cpu_count
from constant import DEFINED, KEYWORD, WHITE_LIST


def _get_var_from_defined(tokens):
    defin = list(set([x for x in tokens if x in DEFINED]))
    var_list = []
    var_str = ''
    if defin:
        index = tokens.index(defin[0])
        try:
            var_list.append(utils.remove_symbol(tokens[index + 1]))
        except IndexError:
            pass
    return [x for x in var_list if x not in KEYWORD + [None, '', ' ']]


def _get_var_from_regexp(tokens):
    var_list = []
    for tok in tokens:
        rep = re.search(r'^\w{1,10}_\w{1,10}[^(]$', tok)
        if rep:
            var_list.append(utils.remove_symbol(rep.group(0)))
    return list(
        filter(
            lambda x: x not in KEYWORD + DEFINED + [None, '', ' '] and x.find('(') == -1,
            var_list))


def _get_func_from_regxp(tokens):
    func_list = []
    for tok in tokens:
        rep = re.search(r"^CWE[\w:~]*\(?\)?$", tok)
        if rep:
            func_list.append(utils.remove_symbol(rep.group(0)))
    return func_list


def _get_var_func(codes):
    var_list = []
    func_list = []
    for line in codes:
        tokens = utils.line_split(line)
        var_list += _get_var_from_defined(tokens) + _get_var_from_regexp(tokens)
        func_list += _get_func_from_regxp(tokens)
    return var_list, func_list


def _var_replace(codes, var_list, func_list):
    syms = []
    var_list.sort()
    for line in codes:
        tokens = utils.line_split_plus(line)
        for k, v in enumerate(tokens):
            if v in var_list:
                tokens[k] = "VAR" + str(var_list.index(v))
            if v in func_list:
                tokens[k] = "FUNC" + str(func_list.index(v))
        syms.append(' '.join(tokens))
    assert len(syms) > 0
    return syms


def symbolize(ps_set):
    sym_set = []
    for codes, label in ps_set:
        var_list, func_list = _get_var_func(codes)
        syms = _var_replace(codes, var_list, func_list)
        sym_set.append([syms, label])
    return sym_set


def sym_save(sym_set):
    with open('../Cache/sym_set.txt', 'a') as f:
        for syms, label in sym_set:
            for line in syms:
                f.write(line + '\n')
            f.write(label + '\n')
            f.write('-----\n')
    logger.debug("sym save success")


def _word_model_train(sym_set):
    sentence_corpus = [y.split(' ') for x in sym_set for y in x[0]]
    if os.path.exists('words.model'):
        model = gensim.models.Word2Vec.load('words.model')
        model.build_vocab(sentence_corpus, update=True)
        model.train(
            sentence_corpus,
            total_examples=model.corpus_count,
            epochs=model.iter)
        logger.info('train words model success')
    else:
        model = gensim.models.Word2Vec(
            sentence_corpus,
            size=100,
            min_count=0,
            workers=cpu_count(),
            iter=5)
        logger.info('create words model success')
    model.save('words.model')
    logger.info('save model success')


def vectorize(sym_set):
    spl_set = []
    x_set = []
    y_set = []
    T = 400
    # split syms
    for syms, label in sym_set:
        spl = [x.split(' ') for x in syms]
        #label = utils.one_hot_embedding(label, 2)
        spl_set.append([spl, label])
    model = gensim.models.Word2Vec.load('words.model')
    # get vec
    for spl, label in spl_set:
        vec_set = [y for x in spl for y in model[x]]
        if len(vec_set) > T:
            x_set.append(vec_set[:T])
        else:
            pad = np.asarray([[0 for x in range(100)]
                              for y in range(T - len(vec_set))])
            vec_set.extend(pad)
            assert len(vec_set) == T
            x_set.append(vec_set)
        y_set.append(int(label))
    X = np.asarray(x_set)
    Y = np.asarray(y_set)
    assert X.shape[0] == Y.shape[0]
    return X, Y


def _ps_loader(file):
    raw_set = []
    with open(file) as f:
        raw = []
        for line in f:
            if line[:-1] != '-' * 33:
                raw.append(line[:-1])
            else:
                raw_set.append(raw)
                raw = []
    return [[x[1:-1], x[-1]] for x in raw_set]


def prep_sym(file, type_):
    if os.path.exists('../Cache/sym_set.txt'):
        return 0
    if type_ == 'ps':
        raw = _ps_loader(file)
    else:
        logger.warn("worry file type")
    sym_set = symbolize(raw)
    sym_save(sym_set)
    logger.info("symbolic dataset create success")


def words_model_training(sym_set):
    sent = [y.split(' ') for x in sym_set for y in x[0]]
    if os.path.exists('words.model'):
        model = gensim.models.Word2Vec.load('words.model')
        model.build_vocab(sent, update=True)
        model.train(
            sent, total_examples=model.corpus_count, epochs=model.epochs)
        logger.info("Update words model success")
    else:
        model = gensim.models.Word2Vec(
            sent, size=100, min_count=0, workers=cpu_count())
        logger.info("Create words model success")
    model.save('words.model')


def prep_wm():
    sym_set = []
    data_config = configer.getData()
    load_rate = data_config['load_rate']
    with open('../Cache/sym_set.txt') as f:
        sym = []
        for line in f:
            if line[:-1] != '-----':
                sym.append(line[:-1])
            else:
                sym_set.append([sym[:-1], sym[-1]])
                sym = []
            if len(sym_set) == load_rate:
                words_model_training(sym_set)
                sym_set = []
        words_model_training(sym_set)


def prep_vec():
    sym_set = []
    data_config = configer.getData()
    load_rate = data_config['load_rate']
    num = 0
    with open('../Cache/sym_set.txt') as f:
        sym = []
        for line in f:
            if line[:-1] != '-----':
                sym.append(line[:-1])
            else:
                sym_set.append([sym[:-1], sym[-1]])
                sym = []
            if len(sym_set) == load_rate:
                X, Y = vectorize(sym_set)
                np.savez("../Cache/dataset{}.npz".format(num), X, Y)
                logger.info("save {} in dataset{} success".format(
                    len(sym_set), num))
                num += 1
                sym_set = []
        X, Y = vectorize(sym_set)
        np.savez("../Cache/dataset{}.npz".format(num), X, Y)
        logger.info("save {} in dataset{} success".format(len(sym_set), num))
