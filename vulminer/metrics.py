#!/usr/bin/env python3
#coding: utf-8

from keras import backend as K


def tp(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

def precision(y_true, y_pred):
    """Precision metric.
     Only computes a batch-wise average of precision.
     Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def recall(y_true, y_pred):
    """Recall metric.
     Only computes a batch-wise average of recall.
     Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    return (2*precision(y_true, y_pred)*recall(y_true, y_pred))/(precision(y_true, y_pred)+recall(y_true, y_pred))


#def tp(y_true, y_pred):
#    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
#    y_pred_neg = 1 - y_pred_pos
#    y_pos = K.round(K.clip(y_true, 0, 1))
#    y_neg = 1 - y_pos
#    return K.sum(y_pos * y_pred_pos)
#
#def tn(y_true, y_pred):
#    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
#    y_pred_neg = 1 - y_pred_pos
#    y_pos = K.round(K.clip(y_true, 0, 1))
#    y_neg = 1 - y_pos
#    return K.sum(y_neg * y_pred_neg)
#
#def fp(y_true, y_pred):
#    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
#    y_pred_neg = 1 - y_pred_pos
#    y_pos = K.round(K.clip(y_true, 0, 1))
#    y_neg = 1 - y_pos
#    return K.sum(y_neg * y_pred_pos)
#
#def fn(y_true, y_pred):
#    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
#    y_pred_neg = 1 - y_pred_pos
#    y_pos = K.round(K.clip(y_true, 0, 1))
#    y_neg = 1 - y_pos
#    return K.sum(y_pos * y_pred_neg)
#
#def p(y_true, y_pred):
#    return tp(y_true, y_pred) + fn(y_true, y_pred)
#
#def n(y_true, y_pred):
#    return tn(y_true, y_pred) + fp(y_true, y_pred)
#
#def tpr(y_true, y_pred):
#    return tp(y_true, y_pred) / (p(y_true, y_pred) + K.epsilon())
#
#def tnr(y_true, y_pred):
#    return tn(y_true, y_pred) / (n(y_true, y_pred) + K.epsilon())
#
#def fnr(y_true, y_pred):
#    return fn(y_true, y_pred) / (p(y_true, y_pred) + K.epsilon())
#
#def fpr(y_true, y_pred):
#    return fp(y_true, y_pred) / (n(y_true, y_pred) + K.epsilon())
#
#def precision(y_true, y_pred):
#    return tp(y_true, y_pred) / (tp(y_true, y_pred) + fp(y_true, y_pred) + K.epsilon())
#
#def f1(y_true, y_pred):
#    return ((2 * precision(y_true, y_pred) * tpr(y_true, y_pred)) / (precision(y_true, y_pred) + tpr(y_true, y_pred) + K.epsilon()))
#
