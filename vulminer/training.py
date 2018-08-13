#!/usr/bin/env python3
#coding: utf-8

import numpy as np
import keras
from keras.layers import LSTM, Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam


class Trainer:
    def __init__(self):
        self.x_train = None
        self.y_train = None

    def input_init(self, vec_set, lsize, tsize):
        self.lsize = lsize
        self.tsize = tsize
        tmp_x= []
        tmp_y = []
        for cgd in vec_set[:lsize+tsize]:
            tmp_x.append(cgd['vector'])
            tmp_y.append(cgd['label'])
        self.x_train = np.array(tmp_x)
        self.y_train = np.array(tmp_y)

    def training(self):
        learning_rate = 0.001
        epochs = 5
        batch_size = 1

        timesteps = 300
        input_dim = 100
        n_hidden = 128
        n_classes = 2

        self.x_train = self.x_train.reshape(-1, timesteps, input_dim)
        print(self.x_train.shape)
        self.y_train = keras.utils.to_categorical(self.y_train, n_classes)
        print(self.y_train.shape)

        model = Sequential()
        input_shape = self.x_train[0].shape
        model.add(
                LSTM(
                    units = n_hidden,
                    return_sequences = True,
                    input_shape=input_shape
            ))
        model.add(Flatten())
        model.add(Dense(units = n_classes, activation = 'tanh'))

        adam = Adam(lr=learning_rate)
        model.summary()
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(self.x_train[:self.lsize], self.y_train[:self.lsize], batch_size=batch_size, epochs=epochs, verbose=1, validation_steps=None)

        scores = model.evaluate(self.x_train[self.size:], self.y_train[self.size:], batch_size=batch_size, verbose=1)

        print('LSTM test accuracy: ', scores[1])
