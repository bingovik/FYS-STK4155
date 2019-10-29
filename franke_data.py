from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from random import random, seed

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import sklearn.linear_model as skl

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.utils import resample

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def CreateDesignMatrix_X(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = x**(i-k) * y**k
    return X

n_x = 100   # number of points
x = np.linspace(0, 1, n_x)
y = np.linspace(0, 1, n_x)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x,y)

# Transform from matrices to vectors
x_1 = np.ravel(x)
y_1 = np.ravel(y)
n = int(len(x_1))
z_true = np.ravel(z)
z_1 = np.ravel(z) + np.random.normal(size=n) * 0.5

m = 5  # degree of polynomial
X = CreateDesignMatrix_X(x_1, y_1, n=m)

X_train, X_test, z_train, z_test, z_true_train, z_true_test = train_test_split(X, z_1, z_true, test_size=0.2)

def MSE(z_data, z_model):
    n = np.size(z_model)
    return np.sum((z_data-z_model)**2)/n

def R2(z_data, z_model):
    return 1 - np.sum((z_data - z_model) ** 2) / np.sum((z_data - np.mean(z_model)) ** 2)


import regression
nn = regression.Neural_TensorFlow()

"""
nn_reg_skl = regression.Neural_TensorFlow()
nn_reg_skl.fit(X_train, z_train)
pred = nn_reg_skl.predict(X_test)
"""


regressor1 = KerasRegressor(build_fn=nn.build_network)  #, epochs=20, batch_size=3)
regressor1.fit(X_train, z_train)
y_pred1 = regressor1.predict(X_test)

### ADD GRIDSEARCH #### 

mse_krr = mean_squared_error(z_test, y_pred1)
print('mse_nn:', mse_krr)
print('mse_nn_1:', MSE(z_true_test, y_pred1))
print('R2_nn:', R2(z_true_test, y_pred1))




##### REGRESSION
beta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(z_train)
ztilde = X_train @ beta

zpredict = X_test @ beta

print("Training MSE: %0.4f" % MSE(z_train, ztilde))
print("Test MSE: %0.4f" % MSE(z_test, zpredict))
print("True MSE: %0.4f" % MSE(z_true_test, zpredict))

print("Training R2: %0.4f" % R2(z_train, ztilde))
print("Test R2: %0.4f" % R2(z_test, zpredict))
print("True R2: %0.4f" % R2(z_true_test, zpredict))
