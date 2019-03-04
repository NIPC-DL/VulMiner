# -*- coding: utf-8 -*-
"""
metrics.py - The collection of metrics

:Author: Verf
:Email: verf@protonmail.com
:License: MIT
"""

def stat(preds, labels):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for p, l in zip(preds, labels):
        if p == l == 1:
            tp += 1
        elif p == l == 0:
            tn += 1
        elif p == 1 and l == 0:
            fp += 1
        else:
            fn += 1
    return tp, tn, fp, fn

def accurary(preds, labels):
    tp, tn, fp, fn = stat(preds, labels)
    try:
        res = (tp+tn)/len(preds)
    except Exception:
        res = -1
    return res

def precision(preds, labels):
    tp, tn, fp, fn = stat(preds, labels)
    try:
        res = tp/(tp+fp)
    except Exception:
        res = -1
    return res

def recall(preds, labels):
    tp, tn, fp, fn = stat(preds, labels)
    try:
        res = tp/(tp+fn)
    except Exception:
        res = -1
    return res

def miss(preds, labels):
    tp, tn, fp, fn = stat(preds, labels)
    try:
        res = fn/(tp+fn)
    except Exception:
        res = -1
    return res

def tpr(preds, labels):
    return recall(preds, labels)

def fnr(preds, labels):
    return miss(preds, labels)

def fpr(preds, labels):
    tp, tn, fp, fn = stat(preds, labels)
    try:
        res = fp/(fp+tn)
    except Exception:
        res = -1
    return res

def f1(preds, labels):
    tp, tn, fp, fn = stat(preds, labels)
    try:
        res = (2*tp)/(2*tp + fp + fn)
    except Exception:
        res = -1
    return res

