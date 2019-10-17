# Data prep
import pandas as pd
import os
import numpy as np
import random
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import matplotlib.patches
from project2_functions import *


class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_neurons=50,
            n_categories=10,
            epochs=40,
            batch_size=100,
            eta=0.1,
            lmbd=0.0):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        # feed-forward for training
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = sigmoid(self.z_h)

        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias

        exp_term = np.exp(self.z_o)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def feed_forward_out(self, X):
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = sigmoid(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        
        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities

    def backpropagation(self):
        error_output = self.probabilities - self.Y_data
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)

        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()






##############################
######## LOGISTIC REGRESSION

class logReg_scikit:

    #def __init__(self)

    def build_model(self):
        model = LogisticRegression(fit_intercept = False)
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


class logisticRegression:

	def __init__(self, _lambda = 0, alpha = 0.1):
		self._lambda = _lambda; self.alpha = alpha;

	def fit(self, X, y, max_iter = 1000):
		_lambda = self._lambda; alpha = self.alpha
		self.beta = np.zeros(X.shape[1])[:,None]
		n = X.shape[0]
		for i in range(max_iter):
			y_pred = sigmoid(X@self.beta)
			grad = X.T@(y_pred - y)/n + np.vstack((0,_lambda*self.beta[1:]/n))
			self.beta = self.beta - alpha*grad

	def train_track_test(self, X, y, X_test, y_test, max_iter = 100, plot = False, savefig = False, filename = ''):
		_lambda = self._lambda; alpha = self.alpha
		self.beta = np.zeros(X.shape[1])[:,None]
		n = X.shape[0]
		cross_entropy_train = np.zeros(max_iter)
		cross_entropy_test = np.zeros(max_iter)
		for i in range(max_iter):
			y_pred = sigmoid(X@self.beta)
			y_pred_test = sigmoid(X_test@self.beta)
			cross_entropy_train[i] = categorical_cross_entropy(y_pred,y)
			cross_entropy_test[i] = categorical_cross_entropy(y_pred_test,y_test)
			#print(categorical_cross_entropy(y_pred,y),categorical_cross_entropy(y_pred_test,y_test))
			grad = X.T@(y_pred - y)/n + np.vstack((0,_lambda*self.beta[1:]/n))
			self.beta = self.beta - alpha*grad
		if plot:
			plot_several(np.repeat(np.arange(max_iter)[:,None], 2, axis=1),
				np.hstack((cross_entropy_train[:,None],cross_entropy_test[:,None])),
				['r-', 'b-'], ['train', 'test'],
				'iterations', 'cross entropy', 'Cross entropy during training',
				savefig = savefig, figname = filename)

	def get_proba(self, X):
		y_pred = sigmoid(X@self.beta)
		return y_pred

	def predict(self, X):
		y_pred_outcome = sigmoid(X@self.beta)
		y_pred_outcome[y_pred_outcome >= 0.5] = 1
		y_pred_outcome[y_pred_outcome < 0.5] = 0
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