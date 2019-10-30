# Data prep
import pandas as pd
import os
import numpy as np
import random
import seaborn as sns
import sys
import os
import keras
from operator import add

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin

import matplotlib.pyplot as plt
import matplotlib.patches

from project2_functions import *

from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from keras import regularizers
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


class Neural_TensorFlow(BaseEstimator, ClassifierMixin):

    def __init__(self, layer_sizes= [100,20],
                batch_size=10,
                epochs=50,
                optimizer="Adam",
                loss="mean_squared_error",
                _lambda = 0,
                activation_function = 'relu'):
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self._lambda = _lambda
        self.activation_function = activation_function

    def build_network(self):
        model = Sequential()
        model.add(BatchNormalization())
        for layer_size in self.layer_sizes:
            model.add(Dense(layer_size, input_dim = 21, activation=self.activation_function, kernel_regularizer=regularizers.l2(self._lambda)))
        model.add(Dense(1, activation = None))
        model.compile(loss=self.loss,
                      optimizer=self.optimizer)
        return model

    def fit(self, X, y):
        self.model = self.build_network(X, y)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

    def predict(self, Xtest):
        return self.model.predict(Xtest)

    def predict_classes(self, Xtest):
        return self.model.predict_classes(Xtest)

    def accuracyscore(self, Xtest, ytest):
        return self.model.evaluate(Xtest, ytest)
        print(accuracy)
