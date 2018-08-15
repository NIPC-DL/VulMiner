#!/usr/bin/env python3
#coding: utf-8

from keras import backend as K


def tp(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    return K.sum(y_pos * y_pred_pos)

def tn(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    return K.sum(y_neg * y_pred_neg)

def fp(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    return K.sum(y_neg * y_pred_pos)

def fn(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    return K.sum(y_pos * y_pred_neg)

def p(y_true, y_pred):
    return tp(y_true, y_pred) + fn(y_true, y_pred)

def n(y_true, y_pred):
    return tn(y_true, y_pred) + fp(y_true, y_pred)

def tpr(y_true, y_pred):
    return tp(y_true, y_pred) / (p(y_true, y_pred) + K.epsilon())

def tnr(y_true, y_pred):
    return tn(y_true, y_pred) / (n(y_true, y_pred) + K.epsilon())

def fnr(y_true, y_pred):
    return fn(y_true, y_pred) / (p(y_true, y_pred) + K.epsilon())

def fpr(y_true, y_pred):
    return fp(y_true, y_pred) / (n(y_true, y_pred) + K.epsilon())

def precision(y_true, y_pred):
    return tp(y_true, y_pred) / (tp(y_true, y_pred) + fp(y_true, y_pred) + K.epsilon())

def f1(y_true, y_pred):
    return ((2 * precision(y_true, y_pred) * tpr(y_true, y_pred)) / (precision(y_true, y_pred) + tpr(y_true, y_pred) + K.epsilon()))

