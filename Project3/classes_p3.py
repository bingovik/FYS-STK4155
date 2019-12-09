from warnings import simplefilter
simplefilter(action = 'ignore', category = FutureWarning)

import pandas as pd
import numpy as np
import seaborn as sns
import keras
import tensorflow as tf
import pdb

from keras.models import Model, Sequential
from keras.layers import Dense
from keras import regularizers

def build_network(layer_sizes=[128, 64, 32],
                n_outputs = 2,
                batch_size=32,
                epochs=20,
                optimizer="Adam",
                loss="categorical_crossentropy",
                alpha = 0,
                activation_function = 'relu',
                output_activation = 'softmax',
                ):
        model = Sequential()
        if isinstance(layer_sizes, int):
            layer_sizes = [layer_sizes]
        for layer_size in layer_sizes:
            model.add(Dense(layer_size, activation=activation_function,kernel_regularizer=regularizers.l2(alpha)))
        model.add(Dense(n_outputs, activation=output_activation))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        return model

class NNclassifier():
    def __init__(self, layer_sizes= [128,64,32],
                n_outputs=2,
                batch_size=32,
                epochs=20,
                optimizer="Adam",
                loss="categorical_crossentropy",
                alpha = 0,
                activation_function = 'relu',
                output_activation = 'softmax'
                ):
        self.layer_sizes = layer_sizes
        self.n_outputs = n_outputs
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.alpha = alpha
        self.activation_function = activation_function
        self.output_activation = output_activation

    def build_network(self):
        model = Sequential()
        #if isinstance(layer_sizes, int):
        #    layer_sizes = [layer_sizes]
        for layer_size in self.layer_sizes:
            model.add(Dense(layer_size, activation = self.activation_function, kernel_regularizer=regularizers.l2(self.alpha)))
        model.add(Dense(10, activation = self.output_activation))
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics = ['accuracy'])
        return model

    def fit(self, X, y):
        self.model = self.build_network()
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def predict(self, Xtest):
        return self.model.predict_classes(Xtest)

    def predict_classes(self, Xtest):
        return self.model.predict_classes(Xtest)

    def accuracyscore(self, Xtest, ytest):
        return self.model.evaluate(Xtest, ytest)
        print(accuracy)

class SVMclassifier():
    def __init__():
        None

    def build(self):
        model = SVC(kernel = 'linear')
        return model

    def fit(self, X, y):
        self.model = self.build(X,y)
        self.model.fit(X, y)


class NNregressor():
    def __init__(self, layer_sizes= [128, 64, 32],
                batch_size=32,
                epochs=20,
                optimizer="Adam",
                loss="mean_squared_error",
                alpha=0,
                activation_function='relu',
                len_X='Xtrain'):
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.alpha = alpha
        self.activation_function = activation_function
        self.len_X = len(len_X[0])

    def build_network(self):
        model = Sequential()
        for layer_size in self.layer_sizes:
            model.add(Dense(layer_size, input_dim = self.len_X, activation=self.activation_function, kernel_regularizer=regularizers.l2(self.alpha)))
        model.add(Dense(1, activation = None))
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics = ['mse'])
        return model

    def fit(self, X, y):
        self.model = self.build_network()
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

    def predict(self, Xtest):
        return self.model.predict(Xtest)

    def predict_classes(self, Xtest):
        return self.model.predict_classes(Xtest)
