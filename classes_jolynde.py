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


class NeuralNetwork(BaseEstimator, ClassifierMixin):

    def __init__(self, n_hidden_neurons=50, activation_function='sigmoid', lmbd=0, epochs=10, batch_size=128, eta=0.01):
        self.lmbd = lmbd
        self.epochs = epochs
        self.batch_size = batch_size
        self.eta = eta
        self.activation_function = activation_function
        if isinstance(n_hidden_neurons, int):
            n_hidden_neurons = (n_hidden_neurons,)
        self.n_hidden_neurons = n_hidden_neurons
        self.n_hidden_layers = len(n_hidden_neurons)
        
        #setting the activation function and its derivative of z
        if activation_function == 'relu':
            self.activation = relu
            self.activation_z_derivative = self.relu_z_derivative
        else:
            self.activation = sigmoid
            self.activation_z_derivative = self.sigmoid_z_derivative

    def sigmoid_z_derivative(self, i):
        return self.a[i]*(1-self.a[i])

    def relu_z_derivative(self, i):
        #return 1 if z>0 and keeping dimensions
        return np.reshape(self.z[i]>0,self.z[i].shape).astype(int)

    def create_biases_and_weights(self):
        #initialize list of layer inputs z and activations (output) a
        self.z = [0]*(self.n_hidden_layers + 1)
        self.a = [0]*self.n_hidden_layers
        self.w = [0]*(self.n_hidden_layers + 1)
        self.bias = [0]*(self.n_hidden_layers + 1)

        #Kaiming initialization of weights (scaling normal random dist with sqrt(2/n))
        self.w[0] = np.random.randn(self.n_features, self.n_hidden_neurons[0])*np.sqrt(2/self.n_features)
        self.bias[0] = np.zeros(self.n_hidden_neurons[0]) + 0.01

        for i in range(1,self.n_hidden_layers):
            self.w[i] = np.random.randn(self.n_hidden_neurons[i-1], self.n_hidden_neurons[i])*np.sqrt(2/self.n_hidden_neurons[i-1])
            self.bias[i] = np.zeros(self.n_hidden_neurons[i]) + 0.01

        self.w[self.n_hidden_layers] = np.random.randn(self.n_hidden_neurons[self.n_hidden_layers-1], self.n_categories)*np.sqrt(2/self.n_hidden_neurons[self.n_hidden_layers-1])
        self.bias[self.n_hidden_layers] = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        # feed-forward for training
        
        #calculating first hidden layer input and activation
        self.z[0] = self.X_data@self.w[0] + self.bias[0]
        self.a[0] = self.activation(self.z[0])

        #looping through rest of hidden layers
        for i in range(1,self.n_hidden_layers):
            self.z[i] = self.a[i-1]@self.w[i] + self.bias[i]
            self.a[i] = self.activation(self.z[i])

        #input and activation of output layer
        self.z[self.n_hidden_layers] = self.a[self.n_hidden_layers-1]@self.w[self.n_hidden_layers] + self.bias[self.n_hidden_layers]
        self.probabilities = softmax(self.z[self.n_hidden_layers])

    def feed_forward_out(self, X):
        # feed-forward for output

        #calculating first hidden layer input and activation
        z = [0]*(self.n_hidden_layers + 1)
        a = [0]*self.n_hidden_layers

        #input and activation for the first hidden layer
        z[0] = X@self.w[0] + self.bias[0]
        a[0] = self.activation(z[0])

        #looping through rest of hidden layers
        for i in range(1,self.n_hidden_layers):
            z[i] = a[i-1]@self.w[i] + self.bias[i]
            a[i] = self.activation(z[i])

        #input and activation of output layer
        z[self.n_hidden_layers] = a[self.n_hidden_layers-1]@self.w[self.n_hidden_layers] + self.bias[self.n_hidden_layers]
        probabilities = softmax(z[self.n_hidden_layers])

        return probabilities

    def backpropagation(self):

        #initialize error and gradient lists
        error = [0]*(self.n_hidden_layers + 1)
        w_grad = [0]*(self.n_hidden_layers + 1)
        bias_grad = [0]*(self.n_hidden_layers + 1)

        #calculating error and gradients for the output layer
        error[self.n_hidden_layers] = self.probabilities - self.Y_data #always the case if using cross entropy (and softmax last layer?)
        w_grad[self.n_hidden_layers] = self.a[self.n_hidden_layers-1].T@error[self.n_hidden_layers] + self.lmbd * self.w[self.n_hidden_layers]
        bias_grad[self.n_hidden_layers] = np.sum(error[self.n_hidden_layers], axis = 0)

        #looping back through hidden layers
        for i in range(self.n_hidden_layers-1,0,-1):
            error[i] = error[i+1]@self.w[i+1].T * self.activation_z_derivative(i)
            w_grad[i] = self.a[i-1].T@error[i] + self.lmbd * self.w[i]
            bias_grad[i] = np.sum(error[i], axis = 0)

        #calculating error and gradients for the first hidden layer
        error[0] = error[1]@self.w[1].T * self.activation_z_derivative(0)
        w_grad[0] = self.X_data.T@error[0] + self.lmbd * self.w[0]
        bias_grad[0] = np.sum(error[0], axis = 0)

        #updating weights and bias
        self.w = list(map(add, self.w, [-i*self.eta for i in w_grad]))
        self.bias = list(map(add, self.bias, [-i*self.eta for i in bias_grad]))

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        outcome = np.zeros(probabilities.shape)
        outcome[probabilities>=0.5] = 1
        return outcome

    def predict_proba(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def accuracy(y, y_pred):
        return accuracy_score(y, y_pred)

    def fit(self, X_data, Y_data, X_test = [], y_test = [], plot_learning = False, savefig = False, filename = ''):
        self.X_data_full = X_data
        self.Y_data_full = Y_data
        self.n_categories = Y_data.shape[1]
        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.iterations = self.n_inputs // self.batch_size
        self.create_biases_and_weights()
        acc_train = np.zeros(self.epochs)
        acc_test = np.zeros(self.epochs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                #mini-batch training data
                self.X_data = self.X_data_full[self.batch_size*j:self.batch_size*(j+1)]
                self.Y_data = self.Y_data_full[self.batch_size*j:self.batch_size*(j+1)]

                self.feed_forward()
                self.backpropagation()

            if plot_learning:
                acc_train[i] = accuracy_score(self.Y_data_full, self.predict(self.X_data_full))
                acc_test[i] = accuracy_score(y_test, self.predict(X_test))
                #acc_train[i] = categorical_cross_entropy(self.Y_data_full, self.predict(self.X_data_full))
                #acc_test[i] = categorical_cross_entropy(y_test, self.predict(X_test))

        if plot_learning:
            plot_several(np.repeat(np.arange(1,self.epochs+1)[:,None], 2, axis=1),
                np.hstack((acc_train[:,None],acc_test[:,None])),
                ['r-', 'b-'], ['train', 'test'],
                'Epochs', 'accuracy', 'Accuracy during training',
                savefig = savefig, figname = filename)

        return acc_train, acc_test

class Neural_TensorFlow(BaseEstimator, ClassifierMixin):

    def __init__(self, layer_sizes=[50],
                batch_size=100,
                epochs=10,
                optimizer="Adam",
                loss="binary_crossentropy",
                _lambda = 0.1,
                activation_function = 'relu'):
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self._lambda = _lambda
        self.activation_function = activation_function

    def build_network(self, X, y):
        model = Sequential()
        model.add(BatchNormalization())
        for layer_size in self.layer_sizes:
            model.add(Dense(layer_size, activation=self.activation_function,kernel_regularizer=regularizers.l2(self._lambda)))
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

    def predict_classes(self, Xtest):
        return self.model.predict_classes(Xtest)

    def accuracyscore(self, Xtest, ytest):
        return self.model.evaluate(Xtest, ytest)
        print(accuracy)



##############################
######## LOGISTIC REGRESSION

class logReg_scikit:

    #def __init__(self)

    def build_model(self):
        model = LogisticRegression(fit_intercept = True)
        return model

    def fit(self, X, y):
        #self.model = LogisticRegression()
        self.model = self.build_model()
        self.model.fit(X, y)

    def get_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    def results(self, X, y):
        results = self.fit(X, y)
        return self.results.fit(X,y)


class logisticRegression(BaseEstimator, ClassifierMixin):

    def __init__(self, _lambda = 0, eta = 0.1, max_iter = 1000):
        self._lambda = _lambda; self.eta = eta; self.max_iter = max_iter

    def fit(self, X, y):
        _lambda = self._lambda; eta = self.eta
        X = np.hstack((np.ones(X.shape[0])[:,None], X))
        self.beta = np.zeros(X.shape[1])[:,None]
        n = X.shape[0]
        for i in range(self.max_iter):
            y_pred = sigmoid(X@self.beta)
            grad = X.T@(y_pred - y)/n + np.vstack((0,_lambda*self.beta[1:]/n))
            self.beta = self.beta - eta*grad

    def train_track_test(self, X, y, X_test, y_test, plot = False, savefig = False, filename = ''):
        _lambda = self._lambda; eta = self.eta
        X = np.hstack((np.ones(X.shape[0])[:,None], X))
        X_test = np.hstack((np.ones(X_test.shape[0])[:,None], X))
        self.beta = np.zeros(X.shape[1])[:,None]
        n = X.shape[0]
        cross_entropy_train = np.zeros(self.max_iter)
        cross_entropy_test = np.zeros(self.max_iter)
        for i in range(self.max_iter):
            y_pred = sigmoid(X@self.beta)
            y_pred_test = sigmoid(X_test@self.beta)
            cross_entropy_train[i] = categorical_cross_entropy(y_pred,y)
            cross_entropy_test[i] = categorical_cross_entropy(y_pred_test,y_test)
            #print(categorical_cross_entropy(y_pred,y),categorical_cross_entropy(y_pred_test,y_test))
            grad = X.T@(y_pred - y)/n + np.vstack((0,_lambda*self.beta[1:]/n))
            self.beta = self.beta - eta*grad
        if plot:
            plot_several(np.repeat(np.arange(self.max_iter)[:,None], 2, axis=1),
                np.hstack((cross_entropy_train[:,None],cross_entropy_test[:,None])),
                ['r-', 'b-'], ['train', 'test'],
                'iterations', 'cross entropy', 'Cross entropy during training',
                savefig = savefig, figname = filename)

    def get_proba(self, X):
        X = np.hstack((np.ones(X.shape[0])[:,None], X))
        y_pred = sigmoid(X@self.beta)
        return y_pred

    def predict(self, X):
        y_pred = self.get_proba(X)
        y_pred_outcome = np.zeros(y_pred.shape)
        y_pred_outcome[y_pred>=0.5] = 1
        return y_pred_outcome

"""
class logReg:
    def __init__(self, _lambda, alpha):
        self._lambda = _lambda
        self.alpha = alpha
    def build_model(self):
        model =
        return model
    def fit(self, X, y):
        self.model = self.build_model()
        self.model.fit(X, y)
    def get_proba(self, X):
        return
    def predict(self, X):
        return
"""
