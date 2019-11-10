import pandas as pd
import os
import numpy as np
import random
import seaborn as sns
import pdb

from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from project2_functions import *
from classes_NN_regression import *

sns.set()

# Creating the Franke data
poly_degree = 7
seed = 0
x = np.sort(np.random.rand(100))
y = np.sort(np.random.rand(100))
xx, yy = np.meshgrid(x,y)
zz_ = FrankeFunction(xx,yy)

sigma = 0.1 #random noise std dev
noise = np.resize(np.random.normal(0, sigma, xx.size),(len(x),len(y)))
zz = zz_ + noise

list_of_features = []
z = np.ravel(zz)
z_true = np.ravel(zz_)
X_orig = np.vstack((np.ravel(xx), np.ravel(yy))).T

# Generate design matrix of polynomials up chosen polynomial degree
poly = PolynomialFeatures(poly_degree) #inlude bias = false
X = poly.fit_transform(X_orig)
features = poly.get_feature_names(['x','y'])
list_of_features.append(features)

# Split and scale data
seed = 0
X_train, X_test, z_train, z_test, z_true_train, z_true_test = train_test_split(X,z,z_true, test_size = 0.2)
sc = StandardScaler()
X_train[:,1:] = sc.fit_transform(X_train[:,1:])
X_test[:,1:] = sc.transform(X_test[:,1:])
#also scaling full set using training data means,stdevs.. Should be ok as the distributions should be very similar
X[:,1:] = sc.transform(X[:,1:]) 

#------------------------------------#

# Set up initial parameters
layer_size = (100,20)
epochs = 150
batch_size = 16
eta = 0.01
lambda_ = 0
activation_function = 'relu'

#create initial test models, own network and Keras/Tensorflow
nn_keras = Neural_TensorFlow(layer_sizes = layer_size, activation_function = activation_function, alpha = lambda_, len_X = X_train)
nn = NeuralNetworkRegressor(n_hidden_neurons = layer_size, activation_function = activation_function, eta = eta, epochs = epochs, batch_size=batch_size)

# Make sure cross_validate uses randomly splitted data
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

#Own network initial run with cross validation on whole data set
cv_nn2 = cross_validate(nn, X, z[:,None], cv = cv, n_jobs = -1, scoring = ('neg_mean_squared_error', 'r2'), return_train_score = False)
print('cv_scores_nn_test:', cv_nn2['test_neg_mean_squared_error'])
print('MSE NN test: %0.5f (+/- %0.5f)' % (-cv_nn2['test_neg_mean_squared_error'].mean(), cv_nn2['test_neg_mean_squared_error'].std()*2))
print('R2 NN test: %0.5f (+/- %0.5f)' % (cv_nn2['test_r2'].mean(), cv_nn2['test_r2'].std()*2))
pdb.set_trace()
#Keras/Tensorflow network initial run with cross validation on whole data set
regressor_keras = KerasRegressor(build_fn=nn_keras.build_network, epochs=epochs, batch_size=batch_size, verbose = 0)
cv_keras = cross_validate(regressor_keras, X, z, cv = cv, scoring = ('neg_mean_squared_error', 'r2'), return_train_score = False, verbose = 0, n_jobs = -1)
print('cv_scores_keras:', cv_keras['test_neg_mean_squared_error'])
print('MSE keras test: %0.5f (+/- %0.5f)' % (-np.mean(cv_keras['test_neg_mean_squared_error']), np.std(cv_keras['test_neg_mean_squared_error'])*2))
print('R2 keras test: %0.5f (+/- %0.5f)' % (np.mean(cv_keras['test_r2']), np.std(cv_keras['test_r2'])*2))

# OLS REGRESSION (to compare with neural networks)
k = 5
mse_train, mse_test, r2_train, r2_test = cross_validation_OLS(X, z, k)
print("cv_MSE_OLS test: %0.5f (+/- %0.5f)" % (mse_test.mean(), mse_test.std()*2))
print("cv_R2_OLS test: %0.5f (+/- %0.5f)" % (r2_test.mean(), r2_test.std()*2))

#run initial NN tests over different learning rates and regularization (with cross validation)
eta_vals = [0.000001, 0.0001, 0.005, 0.01, 0.05]
lmbd_vals = [0, 0.00001, 0.0001, 0.01, 0.1]
scores = np.zeros((len(eta_vals), len(lmbd_vals)))
test_MSE = np.zeros((len(eta_vals), len(lmbd_vals)))
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        DNN_ = NeuralNetworkRegressor(activation_function = activation_function, n_hidden_neurons = layer_size, epochs = epochs,
            batch_size = batch_size, eta=eta, lmbd=lmbd)
        cv_DNN_ = cross_validate(DNN_, X, z[:,None], cv = cv, scoring = 'neg_mean_squared_error', return_train_score = False, n_jobs = -1)
        scores = -np.mean(cv_DNN_['test_score'])
        test_MSE[i][j] = -np.mean(cv_DNN_['test_score'])
        print("Learning rate = ", eta)
        print("Lambda = ", lmbd)
        print('Test MSE:', scores)
        #DNN_.fit(X_train, z_train[:,None], X_test = X_test, y_test = z_test[:,None])
        #scores = MSE(z_true_test[:,None], DNN_.predict(X_test))
        #train_MSE[i][j] = MSE(z_train[:,None], DNN_.predict(X_train))
        #test_true_MSE[i][j] = MSE(z_true_test[:,None], DNN_.predict(X_test))

# plot heatmap of MSE from grid search over eta and lambda
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_MSE, annot=True, ax=ax, cmap="viridis", fmt = '.5g')
ax.set_xticklabels(lmbd_vals)
ax.set_yticklabels(eta_vals)
ax.set_title("MSE test data")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
fig.savefig('./Images/NN_regression1.png')
plt.show()

#best learning rate and regularization parameter (hard coded)
eta_n = 0.01
lambda_n = 1e-5

#GridSearchCV for neural network
parameters = {'n_hidden_neurons':((10,),(100,),(100,20),(128,64,32)), 'activation_function':['sigmoid', 'relu'], 'batch_size':[16,32,64]}
nn = NeuralNetworkRegressor(eta = eta_n, lmbd = lambda_n)
clf = GridSearchCV(nn, parameters, scoring = 'neg_mean_squared_error', cv=cv, verbose = 0)
clf.fit(X_train, z_train[:,None])

print(clf.best_params_)
nnBest = NeuralNetworkRegressor(**clf.best_params_, eta = eta_n, lmbd = lambda_n)
cv_nn_best = cross_validate(nnBest, X, z[:,None], cv = cv, scoring = ('neg_mean_squared_error', 'r2'), return_train_score = False)
print('cv_scores_nn_best_test:', cv_nn_best['test_neg_mean_squared_error'])
print('MSE NN_best test: %0.5f (+/- %0.5f)' % (-cv_nn_best['test_neg_mean_squared_error'].mean(), cv_nn_best['test_neg_mean_squared_error'].std()*2))
print('R2 NN_best test: %0.5f (+/- %0.5f)' % (cv_nn_best['test_r2'].mean(), cv_nn_best['test_r2'].std()*2))

#GridSearchCV on Tensorflow/Keras neural network
regressor_keras_gs = KerasRegressor(build_fn=build_network, loss = 'mean_squared_error', output_activation = None, alpha = lambda_n, n_outputs = 1, verbose = 0)

parameters = {'layer_sizes':((10,),(100,),(100,20), (128,64,32)), 'activation_function':['sigmoid', 'relu'], 'batch_size':[16,32,64]}
clf = GridSearchCV(regressor_keras_gs, parameters, scoring = 'neg_mean_squared_error', cv=cv, verbose = 0, n_jobs=-1)
clf.fit(X_train, z_train)
df_grid_nn_Keras = pd.DataFrame.from_dict(clf.cv_results_)
print(clf.best_params_)
nnKerasBest = KerasRegressor(build_fn=build_network, loss = 'mean_squared_error', output_activation = None, alpha = lambda_n, n_outputs = 1, verbose = 0)
nnKerasBest.set_params(**clf.best_params_)
cv_nn_keras_best = cross_validate(nnKerasBest, X, z[:,None], cv = cv, scoring = ('neg_mean_squared_error', 'r2'), return_train_score = False, verbose = 0)
print('cv_scores_nn_keras_best_test:', cv_nn_keras_best['test_neg_mean_squared_error'])
print('MSE NN_keras_best test: %0.5f (+/- %0.5f)' % (-cv_nn_keras_best['test_neg_mean_squared_error'].mean(), cv_nn_keras_best['test_neg_mean_squared_error'].std()*2))
print('R2 NN_keras_best test: %0.5f (+/- %0.5f)' % (cv_nn_keras_best['test_r2'].mean(), cv_nn_keras_best['test_r2'].std()*2))


## BEST MODELS

print('_______BEST MODELS_________')
layer_size = (128,64,32)
epochs = 150
batch_size = 32
eta = 0.01
lambda_ = 1e-5
activation_function = 'relu'

nn_keras1 = Neural_TensorFlow(layer_sizes = layer_size, activation_function = activation_function, alpha = lambda_, len_X = X_train)
nn2 = NeuralNetworkRegressor(n_hidden_neurons = layer_size, activation_function = activation_function, eta = eta, lmbd = lambda_, epochs = epochs, batch_size=batch_size)

cv_nn1 = cross_validate(nn2, X, z[:,None], cv = cv, n_jobs = -1, scoring = ('neg_mean_squared_error', 'r2'), return_train_score = False)
print('cv_scores_nn_test:', cv_nn1['test_neg_mean_squared_error'])
print('MSE NN test: %0.5f (+/- %0.5f)' % (-cv_nn1['test_neg_mean_squared_error'].mean(), cv_nn1['test_neg_mean_squared_error'].std()*2))
print('R2 NN test: %0.5f (+/- %0.5f)' % (cv_nn1['test_r2'].mean(), cv_nn1['test_r2'].std()*2))

regressor_keras = KerasRegressor(build_fn=nn_keras1.build_network, epochs=epochs, batch_size=batch_size, verbose = 0)
cv_keras = cross_validate(regressor_keras, X, z, cv = cv, scoring = ('neg_mean_squared_error', 'r2'), return_train_score = False, verbose = 0, n_jobs = -1)
print('cv_scores_keras:', cv_keras['test_neg_mean_squared_error'])
print('MSE keras test: %0.5f (+/- %0.5f)' % (-np.mean(cv_keras['test_neg_mean_squared_error']), np.std(cv_keras['test_neg_mean_squared_error'])*2))
print('R2 keras test: %0.5f (+/- %0.5f)' % (np.mean(cv_keras['test_r2']), np.std(cv_keras['test_r2'])*2))


## And a final check on how they perform on the true data
regressor_keras = KerasRegressor(build_fn=nn_keras1.build_network, epochs=epochs, batch_size=batch_size, verbose = 0)
regressor_keras.fit(X_train, z_train)
keras_pred = regressor_keras.predict(X_test)
print('MSE_keras_test:', mean_squared_error(z_test, keras_pred))
print('MSE_keras_true:', mean_squared_error(z_true_test, keras_pred))
print('R2_keras_test:', R2(z_test, keras_pred))
print('R2_keras_true:', R2(z_true_test, keras_pred))

nn1 = NeuralNetworkRegressor(n_hidden_neurons = layer_size, activation_function = activation_function, eta = eta, lmbd = lambda_, epochs = epochs, batch_size=batch_size)
nn1.fit(X_train, z_train[:,None], X_test = X_test, y_test = z_test[:,None])
y_pred_nn1 = nn1.predict(X_test)
mse_nn1 = mean_squared_error(z_test[:,None], y_pred_nn1)
print('mse_nn_test:', mse_nn1)
print('mse_nn_true:', mean_squared_error(z_true_test[:,None], y_pred_nn1))
print('R2_nn_test:', r2_score(z_test[:,None], y_pred_nn1))
print('R2_nn_true:', r2_score(z_true_test[:,None], y_pred_nn1))

beta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(z_train)
ztilde = X_train @ beta
zpredict = X_test @ beta
print("mse_ols_test:", mean_squared_error(z_test, zpredict))
print("mse_ols_true:", mean_squared_error(z_true_test, zpredict))
print("R2_ols_test:", r2_score(z_test, zpredict))
print("R2_ols_test:", r2_score(z_true_test, zpredict))
