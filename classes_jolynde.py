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
