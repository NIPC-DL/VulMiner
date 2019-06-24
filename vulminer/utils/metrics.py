# -*- coding: utf-8 -*-
"""
metrics.py - The collection of metrics

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""
import sklearn.metrics as skm
from vulminer.utils import logger

def base(labels, preds):
    cm = skm.confusion_matrix(labels, preds)
    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]
    return tp, tn, fp, fn

def fnr(labels, preds):
    tp, tn, fp, fn = base(labels, preds)
    try:
        res = fn/(fn+tp)
    except Exception:
        res = -1
    return res

def fpr(labels, preds):
    tp, tn, fp, fn = base(labels, preds)
    logger.info(f'{tp}, {tn}, {fp}, {fn}')
    try:
        res = fp/(fp+tn)
    except Exception:
        res = -1
    return res
