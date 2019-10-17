from random import choice
import sys
import os
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from keras import regularizers
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import tensorflow as tf


class Neural:

    def __init__(self, layer_sizes=[50],
                batch_size=100,
                epochs=10,
                optimizer="Adam",
                loss="binary_crossentropy",
                alpha = 0.1):
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.alpha = alpha

    def build_network(self, X, y):
        model = Sequential()
        model.add(BatchNormalization())
        for layer_size in self.layer_sizes:
            model.add(Dense(layer_size, activation='relu',kernel_regularizer=regularizers.l2(self.alpha)))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['accuracy'])
        return model

    def fit(self, X, y):
        self.model = self.build_network(X, y)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

    def predict(self, Xtest):
        return self.model.predict(Xtest)

    def accuracyscore(self, Xtest, ytest):
        return self.model.evaluate(Xtest, ytest)
        print(accuracy)
