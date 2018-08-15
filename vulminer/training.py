#!/usr/bin/env python3
#coding: utf-8

import numpy as np
import keras
from logger import logger
from keras.layers import LSTM, Dense, Activation, Flatten, Bidirectional, TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
#from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import KFold
from metrics import f1, fpr, fnr, tpr, precision

class Trainer:
    def __init__(self, config):
        self._config = config

    def load_data(self, x_set, y_set):
        self.x_set = x_set.reshape(-1, int(self._config['Model']['timesteps']), int(self._config['Model']['input_dim']))
        self.y_set = to_categorical(y_set, int(self._config['Model']['n_classes']))
        self.x_shape = x_set.shape
        self.y_shape = y_set.shape
        logger.debug('x_shape: ' + str(self.x_shape))
        logger.debug('y_shape: ' + str(self.y_shape))

    def training(self):
        seed = 7
        np.random.seed(seed)
        k_folds = int(self._config['Model']['kfolds'])
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        model = self._set_model()
        cross_score = []
        num = 0
        for train, test in kf.split([0 for x in range(self.y_set.shape[0])]):
            num  += 1
            logger.info('start {0} folds validation: {1}'.format(str(k_folds), str(num)))
            model.fit(
                    self.x_set[train],
                    self.y_set[train],
                    batch_size=int(self._config['Model']['batch_size']),
                    epochs=int(self._config['Model']['epochs']),
                    verbose=1,
                )
            score = model.evaluate(self.x_set[test], self.y_set[test], verbose=1)
            cross_score.append(score)
            model.summary()
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cross_score), np.std(cross_score)))
        self.model = model


    def _set_model(self):
        model = Sequential()
        logger.debug('input_shape: ' + str(self.x_shape[1:]))
        model.add(Bidirectional(LSTM(
            units=int(self._config['LSTM']['units']),
            activation=self._config['LSTM']['activation'],
            dropout=float(self._config['LSTM']['dropout']),
            return_sequences=True,
            input_shape=self.x_shape[1:]
            )))
        model.add(Flatten())
        model.add(Dense(
            units=int(self._config['Dense']['units']),
            activation=self._config['Dense']['activation']
            ))

        adam = Adam(lr=float(self._config['Model']['learning_rate']))

        model.compile(
                optimizer=adam,
                loss=self._config['Model']['loss'],
                metrics=[fpr, fnr, tpr, precision, f1]
                )
        return model


