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

from operator import add


class NeuralNetwork:

    def __init__(self, n_hidden_neurons=50, activation='sigmoid'):

        if isinstance(n_hidden_neurons, int):
            n_hidden_neurons = (n_hidden_neurons,)
        self.n_hidden_neurons = n_hidden_neurons
        self.n_hidden_layers = len(n_hidden_neurons)
        self.activation = activation

        #setting the activation function and its derivative of z
        if activation == 'relu':
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
        '''
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons[0])
        self.hidden_bias = np.zeros(self.n_hidden_neurons[0]) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons[0], self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01
        '''
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
        '''
        self.z_h = self.X_data@self.hidden_weights + self.hidden_bias
        self.a_h = sigmoid(self.z_h)

        self.z_o = self.a_h@self.output_weights + self.output_bias

        exp_term = np.exp(self.z_o)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        '''
        #initialize list of layer inputs z and activations (output) a
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
        '''
        z_h = X@self.hidden_weights + self.hidden_bias
        a_h = sigmoid(z_h)

        z_o = a_h@self.output_weights + self.output_bias

        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities
        '''
        #initialize list of layer inputs z and activations (output) a
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
        '''
        error_output = self.probabilities - self.Y_data #always the case if using cross entropy (and softmax last layer?)
        error_hidden = error_output@self.output_weights.T * self.a_h * (1 - self.a_h) #last two factors change if using another activation function
        #error_hidden = error_output@self.output_weights.T * sigmoid_derivative(self.z_h) #last two factors change if using another activation function
        #error_hidden = error_output@self.output_weights.T * relu_derivative(self.z_h)
        #error_hidden = error_output@self.output_weights.T * activation_derivative()

        self.output_weights_gradient = self.a_h.T@error_output
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = self.X_data.T@error_hidden
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.outputput_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient
        '''

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
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self, X_data, Y_data, activation = 'sigmoid', epochs=10, batch_size=128, eta=0.1, lmbd=0.0):
        self.X_data_full = X_data
        self.Y_data_full = Y_data
        self.n_categories = Y_data.shape[1]
        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd
        self.create_biases_and_weights()
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                #mini-batch training data
                self.X_data = self.X_data_full[self.batch_size*j:self.batch_size*(j+1)]
                self.Y_data = self.Y_data_full[self.batch_size*j:self.batch_size*(j+1)]

                self.feed_forward()
                self.backpropagation()

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
