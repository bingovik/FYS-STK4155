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
zz_ = FrankeFunction(xx,yy)
sigma = 0.5 #random noise std dev
seed = 0
noise = np.resize(np.random.normal(0, sigma, xx.size),(len(x),len(y)))
zz = zz_ + noise
fignamePostFix = '_Franke_40_01'

#plot data
#surfPlot(xx, yy, zz, savefig = savefigs, figname = 'surfDataRaw' + fignamePostFix)

list_of_features = []

#making useful variables
poly_degrees = np.arange(poly_degree_max)+1
lambda_tests_str = ['%.1E' % Decimal(str(lam)) for lam in lambda_tests]
z = np.ravel(zz) #[:,None]
z_true = np.ravel(zz_) #[:,None]
X_orig = np.vstack((np.ravel(xx), np.ravel(yy))).T

print(z.shape)
print(z_true.shape)

# Generate design matrix of polygons up to with chosen polynomial degree
poly = PolynomialFeatures(poly_degree) #inlude bias = false
X = poly.fit_transform(X_orig)
features = poly.get_feature_names(['x','y'])
list_of_features.append(features)

# Split and scale the data
seed = 0
X_train, X_test, z_train, z_test, z_true_train, z_true_test = train_test_split(X,z,z_true, test_size = 0.2)
sc = StandardScaler()
X_train[:,1:] = sc.fit_transform(X_train[:,1:])
X_test[:,1:] = sc.transform(X_test[:,1:])
X[:,1:] = sc.transform(X[:,1:])

print(X_train.shape)

scikitlearn_OLS = LinearRegression()
scikitlearn_OLS.fit(X_train, z_train)
print(mean_squared_error(z_test, scikitlearn_OLS.predict(X_test)))

#Functions MSE and R2
def MSE(z_data, z_model):
    n = np.size(z_model)
    return np.sum((z_data-z_model)**2)/n

def R2(z_data, z_model):
    return 1 - np.sum((z_data - z_model) ** 2) / np.sum((z_data - np.mean(z_model)) ** 2)


#------------------------------------#

### NEURAL NET REGRESSION
from regression import *
nn_keras = Neural_TensorFlow(layer_sizes = (100,20), activation_function = 'relu', len_X = X_train)
nn = NeuralNetworkRegressor(n_hidden_neurons = (100,20), activation_function = 'relu', eta = 0.01, epochs = 50, batch_size=16)

regressor_keras = KerasRegressor(build_fn=nn_keras.build_network, epochs=50, batch_size=10)
regressor_keras.fit(X_train, z_train)
y_pred_keras = regressor_keras.predict(X_test)

mse_keras = mean_squared_error(z_test, y_pred_keras)
print('mse_keras:', mse_keras)
print('mse_keras_true:', MSE(z_true_test, y_pred_keras))
print('R2_keras_true:', R2(z_true_test, y_pred_keras))

cv_keras = cross_val_score(regressor_keras, X, z, cv = 5, scoring = 'neg_mean_squared_error')
print('cv_scores_keras:', cv_keras)
print('cv_mean_keras:', -np.mean(cv_keras))

nn.fit(X_train, z_train[:,None], X_test = X_test, y_test = z_test[:,None])
y_pred_nn = nn.predict(X_test)

mse_nn = mean_squared_error(z_test[:,None], y_pred_nn)
print('mse_nn:', mse_nn)
print('mse_nn_true:', MSE(z_true_test[:,None], y_pred_nn))
print('R2_nn_true:', R2(z_true_test[:,None], y_pred_nn))

cv_nn = cross_val_score(nn, X, z[:,None], cv = 5, scoring = 'neg_mean_squared_error')
print('cv_scores_nn:', cv_nn)
print('cv_mean_nn:', -np.mean(cv_nn))

##### OLS REGRESSION
beta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(z_train)
ztilde = X_train @ beta

zpredict = X_test @ beta

print("Training MSE: %0.4f" % MSE(z_train, ztilde))
print("Test MSE: %0.4f" % MSE(z_test, zpredict))
print("True MSE: %0.4f" % MSE(z_true_test, zpredict))

print("Training R2: %0.4f" % R2(z_train, ztilde))
print("Test R2: %0.4f" % R2(z_test, zpredict))
print("True R2: %0.4f" % R2(z_true_test, zpredict))


epochs = 50
batch_size = 10
eta_vals = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1] #np.logspace(-2, 1, 7)
lmbd_vals = [0, 1e-5, 1e-3, 0.01, 0.05, 0.1, 1.0] #np.logspace(-2, 1, 7)

import seaborn as sns

sns.set()

DNN_nn = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
scores = np.zeros((len(eta_vals), len(lmbd_vals)))
train_MSE = np.zeros((len(eta_vals), len(lmbd_vals)))
test_true_MSE = np.zeros((len(eta_vals), len(lmbd_vals)))
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        DNN_ = NeuralNetworkRegressor(n_hidden_neurons = (50,), epochs = epochs, batch_size = batch_size,
                                         eta=eta, lmbd=lmbd)
        DNN_.fit(X_train, z_train[:,None], X_test = X_test, y_test = z_test[:,None])
        scores = MSE(z_true_test[:,None], DNN_.predict(X_test))

        DNN_nn[i][j] = DNN_

        #print("Learning rate = ", eta)
        #print("Lambda = ", lmbd)
        #print("Test MSE: %.3f" % scores)
        #print()

        train_MSE[i][j] = MSE(z_train[:,None], DNN_.predict(X_train))
        test_true_MSE[i][j] = MSE(z_true_test[:,None], DNN_.predict(X_test))

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_MSE, annot=True, ax=ax, cmap="viridis")
ax.set_xticklabels(lmbd_vals)
ax.set_yticklabels(eta_vals)
ax.set_title("Training MSE")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_true_MSE, annot=True, ax=ax, cmap="viridis")
ax.set_xticklabels(lmbd_vals)
ax.set_yticklabels(eta_vals)
ax.set_title("MSE on the true data")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

### ADD CROSS VALIDATION ####





"""
train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        DNN = DNN_nn[i][j]

        train_MSE[i][j] = MSE(z_train, DNN.predict(X_train))
        test_MSE[i][j] = MSE(z_test, DNN.predict(X_test))


fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_MSE, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training MSE")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_MSE, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test MSE")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()



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

"""
