import pandas as pd
import os
import numpy as np
import random
import seaborn as sns
import pdb

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, mean_squared_error
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import matplotlib.patches

#import statsmodels.api as sm
from project2_functions import *
import classes_jolynde

from project2_functions import *
import matplotlib.pyplot as plt
from decimal import Decimal
import seaborn as sns
sns.set()

# Creating the Franke data
poly_degree = 7

seed = 0
x = np.sort(np.random.rand(10))
y = np.sort(np.random.rand(10))
xx, yy = np.meshgrid(x,y)
zz_ = FrankeFunction(xx,yy)
sigma = 0.1 #random noise std dev
noise = np.resize(np.random.normal(0, sigma, xx.size),(len(x),len(y)))
zz = zz_ + noise

list_of_features = []
z = np.ravel(zz) #[:,None]
z_true = np.ravel(zz_) #[:,None]
X_orig = np.vstack((np.ravel(xx), np.ravel(yy))).T

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


#------------------------------------#

### NEURAL NET REGRESSION
from regression import *

layer_size = (100,20)
epochs = 50
batch_size = 10
eta = 0.01
lambda_ = 0

nn_keras = Neural_TensorFlow(layer_sizes = layer_size, activation_function = 'sigmoid', len_X = X_train)
nn = NeuralNetworkRegressor(n_hidden_neurons = layer_size, activation_function = 'sigmoid', eta = eta, epochs = epochs, batch_size=batch_size)

"""
nn.fit(X_train, z_train[:,None], X_test = X_test, y_test = z_test[:,None])
y_pred_nn = nn.predict(X_test)

mse_nn = mean_squared_error(z_test[:,None], y_pred_nn)
print('mse_nn:', mse_nn)
print('mse_nn_true:', MSE(z_true_test[:,None], y_pred_nn))
print('R2_nn_true:', R2(z_true_test[:,None], y_pred_nn))
"""

cv_nn2 = cross_validate(nn, X, z[:,None], cv = 5, scoring = ('neg_mean_squared_error', 'r2'), return_train_score = False)
print('cv_scores_nn_test:', cv_nn2['test_neg_mean_squared_error'])
#print('MSE NN train: %0.5f (+/- %0.5f)' % (-cv_nn['train_neg_mean_squared_error'].mean(), cv_nn['train_neg_mean_squared_error'].std()*2))
print('MSE NN test: %0.5f (+/- %0.5f)' % (-cv_nn2['test_neg_mean_squared_error'].mean(), cv_nn2['test_neg_mean_squared_error'].std()*2))
#print('R2 NN train: %0.5f (+/- %0.5f)' % (-cv_nn['train_r2'].mean(), cv_nn['train_r2'].std()*2))
print('R2 NN test: %0.5f (+/- %0.5f)' % (cv_nn2['test_r2'].mean(), cv_nn2['test_r2'].std()*2))

##### OLS REGRESSION
k = 5
mse_train, mse_test, r2_train, r2_test = cross_validation(X, z, k)
print("cv_MSE_OLS train: %0.5f (+/- %0.5f)" % (mse_train.mean(), mse_train.std()*2))
print("cv_MSE_OLS test: %0.5f (+/- %0.5f)" % (mse_test.mean(), mse_test.std()*2))
print("cv_R2_OLS train: %0.5f (+/- %0.5f)" % (r2_train.mean(), r2_train.std()*2))
print("cv_R2_OLS test: %0.5f (+/- %0.5f)" % (r2_test.mean(), r2_test.std()*2))

# Which value for eta & lambda
eta_vals = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1] #np.logspace(-2, 1, 7)
lmbd_vals = [0, 1e-5, 1e-4, 1e-3, 0.01, 0.1] #np.logspace(-2, 1, 7)

DNN_nn = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
scores = np.zeros((len(eta_vals), len(lmbd_vals)))
train_MSE = np.zeros((len(eta_vals), len(lmbd_vals)))
test_MSE = np.zeros((len(eta_vals), len(lmbd_vals)))
test_true_MSE = np.zeros((len(eta_vals), len(lmbd_vals)))
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        DNN_ = NeuralNetworkRegressor(n_hidden_neurons = layer_size, epochs = epochs, batch_size = batch_size,
                                         eta=eta, lmbd=lmbd)
        cv_DNN_ = cross_validate(DNN_, X, z[:,None], cv = 5, scoring = 'neg_mean_squared_error', return_train_score = False)
        scores = -np.mean(cv_DNN_['test_score'])

        test_MSE[i][j] = -np.mean(cv_DNN_['test_score'])
        print("Learning rate = ", eta)
        print("Lambda = ", lmbd)
        print('Test MSE: %.3f' % scores)

        #DNN_.fit(X_train, z_train[:,None], X_test = X_test, y_test = z_test[:,None])
        #scores = MSE(z_true_test[:,None], DNN_.predict(X_test))

        #train_MSE[i][j] = MSE(z_train[:,None], DNN_.predict(X_train))
        #test_true_MSE[i][j] = MSE(z_true_test[:,None], DNN_.predict(X_test))

"""
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_MSE, annot=True, ax=ax, cmap="viridis")
ax.set_xticklabels(lmbd_vals)
ax.set_yticklabels(eta_vals)
ax.set_title("Training MSE")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()
"""
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_MSE, annot=True, ax=ax, cmap="viridis")
ax.set_xticklabels(lmbd_vals)
ax.set_yticklabels(eta_vals)
ax.set_title("MSE test data")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
fig.savefig('./Images/NN_regression1.png')
plt.show()


eta_n = 0.01
lmbd_n = 0

# ADD GRIDSEARCH
parameters = {'n_hidden_neurons':((10,),(100,),(100,20),(128,64,32)), 'activation_function':['sigmoid', 'relu'], 'epochs':[50,100]}
nn = NeuralNetworkRegressor(eta = eta_n, lmbd = lmbd_n)
clf = GridSearchCV(nn, parameters, scoring = 'neg_mean_squared_error', cv=5, verbose = 0)
clf.fit(X, z[:,None])

print(clf.best_params_)
nnBest = NeuralNetworkRegressor(**clf.best_params_, eta = eta_n)
cv_nn_best = cross_validate(nnBest, X, z[:,None], cv = 5, scoring = ('neg_mean_squared_error', 'r2'), return_train_score = False)
print('cv_scores_nn_best_test:', cv_nn_best['test_neg_mean_squared_error'])
print('MSE NN_best test: %0.5f (+/- %0.5f)' % (-cv_nn_best['test_neg_mean_squared_error'].mean(), cv_nn_best['test_neg_mean_squared_error'].std()*2))
print('R2 NN_best test: %0.5f (+/- %0.5f)' % (cv_nn_best['test_r2'].mean(), cv_nn_best['test_r2'].std()*2))


# Compare with Keras
regressor_keras = KerasRegressor(build_fn=nn_keras.build_network, epochs=epochs, batch_size=batch_size)
#regressor_keras.fit(X_train, z_train)
#y_pred_keras = regressor_keras.predict(X_test)

#mse_keras = mean_squared_error(z_test, y_pred_keras)
#print('mse_keras:', mse_keras)
#print('mse_keras_true:', MSE(z_true_test, y_pred_keras))
#print('R2_keras_true:', R2(z_true_test, y_pred_keras))

cv_keras = cross_validate(regressor_keras, X, z, cv = 5, scoring = ('neg_mean_squared_error', 'r2'), return_train_score = True)
print('cv_scores_keras:', cv_keras['test_neg_mean_squared_error'])
print('MSE keras test: %0.5f (+/- %0.5f)' % (-np.mean(cv_keras['test_neg_mean_squared_error']), np.std(cv_keras['test_neg_mean_squared_error'])*2))
print('R2 keras test: %0.5f (+/- %0.5f)' % (-np.mean(cv_keras['test_r2']), np.std(cv_keras['test_r2'])*2))






"""
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
