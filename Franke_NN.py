import pandas as pd
import os
import numpy as np
import random
import seaborn as sns
import pdb

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, mean_squared_error
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression

import matplotlib.pyplot as plt
import matplotlib.patches

#import statsmodels.api as sm
from project2_functions import *
import classes_jolynde

from project2_functions import *
import matplotlib.pyplot as plt
from decimal import Decimal

# setting models and hyper parameters
poly_degree = 5
poly_degree_max = 10
lambda_test_number = 15
lambda_tests = np.logspace(-7, 1, num=lambda_test_number)

savefigs = True

# set noise and number of data points for Franke function evaluation
seed = 0
x = np.sort(np.random.rand(40))
seed = 0
y = np.sort(np.random.rand(40))
xx, yy = np.meshgrid(x,y)
zz = FrankeFunction(xx,yy)
sigma = 0.1 #random noise std dev
seed = 0
noise = np.resize(np.random.normal(0, sigma, xx.size),(len(x),len(y)))
zz = zz + noise
fignamePostFix = '_Franke_40_01'

#plot data
surfPlot(xx, yy, zz, savefig = savefigs, figname = 'surfDataRaw' + fignamePostFix)

list_of_features = []

#making useful variables
poly_degrees = np.arange(poly_degree_max)+1
lambda_tests_str = ['%.1E' % Decimal(str(lam)) for lam in lambda_tests]
z = np.ravel(zz)[:,None]
X_orig = np.vstack((np.ravel(xx), np.ravel(yy))).T

# Generate design matrix of polygons up to with chosen polynomial degree
poly = PolynomialFeatures(poly_degree) #inlude bias = false
X = poly.fit_transform(X_orig)
features = poly.get_feature_names(['x','y'])
list_of_features.append(features)

# Split and scale the data
seed = 0
X_train, X_test, z_train, z_test = train_test_split(X,z, test_size = 0.2)
sc = StandardScaler()
X_train[:,1:] = sc.fit_transform(X_train[:,1:])
X_test[:,1:] = sc.transform(X_test[:,1:])
X[:,1:] = sc.transform(X[:,1:])

scikitlearn_OLS = LinearRegression()
scikitlearn_OLS.fit(X_train, z_train)
print(mean_squared_error(z_test, scikitlearn_OLS.predict(X_test)))

nn = classes_jolynde.NeuralNetworkRegressor(n_hidden_neurons = (100,20), activation_function = 'relu', eta = 0.01, epochs = 500, batch_size=128)
nn.fit(X_train,z_train, X_test = X_test, y_test = z_test, plot_learning = True)

nn = classes_jolynde.NeuralNetworkRegressor(n_hidden_neurons = (40,20), activation_function = 'relu', eta = 0.01, epochs = 500, batch_size=128)
nn.fit(X_train,z_train, X_test = X_test, y_test = z_test, plot_learning = True)

nn = classes_jolynde.NeuralNetworkRegressor(n_hidden_neurons = 100, activation_function = 'relu', eta = 0.01, epochs = 500, batch_size=128)
nn.fit(X_train,z_train, X_test = X_test, y_test = z_test, plot_learning = True)

nn = classes_jolynde.NeuralNetworkRegressor(n_hidden_neurons = (16,8), activation_function = 'relu', eta = 0.01, epochs = 500, batch_size=128)
nn.fit(X_train,z_train, X_test = X_test, y_test = z_test, plot_learning = True)

z_predict = nn.predict(X)
surfPlot(xx, yy, z_predict.reshape((len(y),len(x))), savefig = False, figname = 'Ridge_best_val_Surf' + fignamePostFix)


# running regressions using different polynomial fits up to poly_degree_max
for poly_degree in poly_degrees:
    print('Polynomial degree: %g' % poly_degree)
    # creating polynomials of degree poly_degree
    poly = PolynomialFeatures(poly_degree) #inlude bias = false
    X = poly.fit_transform(X_orig)
    features = poly.get_feature_names(['x','y'])
    list_of_features.append(features)

    # Split and scale the data
    seed = 0
    X_train, X_test, z_train, z_test = train_test_split(X,z, test_size = 0.2)
    sc = StandardScaler()
    X_train[:,1:] = sc.fit_transform(X_train[:,1:])
    X_test[:,1:] = sc.transform(X_test[:,1:])
    X[:,1:] = sc.transform(X[:,1:])

    scikitlearn_OLS = LinearRegression()
    scikitlearn_OLS.fit(X_train, z_train)
    
    nn = classes_jolynde.NeuralNetworkRegressor(n_hidden_neurons = (100,20), activation_function = 'relu', eta = 0.03, epochs = 160, batch_size=128)
    nn.fit(X_train,z_train, X_test = X_test, y_test = z_test, plot_learning = True)
    z_predict = nn.predict(X)
    surfPlot(xx, yy, z_predict.reshape((len(y),len(x))), savefig = False, figname = 'Ridge_best_val_Surf' + fignamePostFix)

pdb.set_trace()