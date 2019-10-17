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

##############################
######## LOGISTIC REGRESSION

class logReg_scikit:

    #def __init__(self)

    def build_model(self):
        model = LogisticRegression()
        return model

    def fit(self, X, y):
        #self.model = LogisticRegression()
        self.model = self.build_model()
        self.model.fit(X, y)

    def get_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)
"""
class logReg:

    def __init__(self, _lambda, alpha):
        self._lambda = _lambda
        self.alpha = alpha

    def cost_function(self, theta, x, y):
        # Computes the cost function for all the training samples
        m = x.shape[0]
        total_cost = -(1 / m) * np.sum(
            y * np.log(probability(theta, x)) + (1 - y) * np.log(
                1 - probability(theta, x)))
        return total_cost

    def gradient(self, theta, x, y):
        # Computes the gradient of the cost function at the point theta
        m = x.shape[0]
        return (1 / m) * np.dot(x.T, sigmoid(net_input(theta, x)) - y)

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



# FUNCTIONS

def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))

def net_input(theta, x):
    # Computes the weighted sum of inputs
    return np.dot(x, theta)

def probabilities(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(net_input(theta, x))
