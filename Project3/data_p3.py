import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, ShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, make_scorer, accuracy_score, confusion_matrix

from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

from sklearn import svm

from classes_p3 import *

dfr = pd.read_csv('./Data/winequality-red.csv', sep = ';')
dfw = pd.read_csv('./Data/winequality-white.csv', sep = ';')

# Create design matrix and target variable
X = dfr.iloc[:, :11].values
y = dfr.iloc[:, 11].values
y_c = dfr.iloc[:, 11]

# Holdout data for final test
X, X_T, y, y_T, y_c, y_c_T = train_test_split(X, y, y_c, test_size = 0.2, random_state = 5)

#One hot encoding, it automatically does the cat/object variables
sc = StandardScaler()

originalclass = []
predictedclass = []
def classification_report_with_accuracy_score(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    return accuracy_score(y_true, y_pred) #return accuracy score

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

from sklearn.metrics import make_scorer
def MAD(y_true, y_pred):
    return (sum(abs(y_true - y_pred)) / len(y_true))

scoring = {'MSE': 'neg_mean_squared_error', 'MAD': make_scorer(MAD, greater_is_better = False), 'R2': 'r2'}

#######
#CATEGORICAL CLASSIFICATION
#######
"""
y_onehot = pd.get_dummies(y_c).values

Xtrain, Xtest, ytrain_onehot, ytest_onehot, ytrain_, ytest_ = train_test_split(X, y_onehot, y, test_size = 0.2, random_state = 5)
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)
X_ = sc.transform(X)

LR = LogisticRegression()
NNc = NNclassifier()
SVMc = svm.SVC(gamma = 'auto')

#### LOGISTIC REGRESSION
reg = LogisticRegression(solver = 'lbfgs', max_iter = 10000, multi_class = 'auto')


#### SUPPORT VECTOR MACHINE
print('-------------SVM------------')
scores_SVMc = cross_val_score(SVMc, X_, y, cv = cv)
#print(scores_SVMc)
print('CV accuracy score SVMc: %0.3f (+/- %0.3f)' % (scores_SVMc.mean(), scores_SVMc.std()*2))


# Find the best values for the hyperparameters of the SVM
C_vals = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4] #[0.01, 0.1, 1, 10, 1e2, 1e4, 1e6]
gamma_vals = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4] #[10, 1, 0.1, 0.01, 0.0001, 1e-5, 1e-7]
scores = np.zeros((len(C_vals), len(gamma_vals)))
test_acc = np.zeros((len(C_vals), len(gamma_vals)))
for i, gamma in enumerate(gamma_vals):
    for j, C in enumerate(C_vals):
        model = svm.SVC(kernel = 'rbf', C = C, gamma = gamma)
        cv_ = cross_val_score(model, X_, y, cv = cv, n_jobs = -1, verbose = 0)
        scores = np.mean(cv_)
        test_acc[i][j] = np.mean(cv_)
        print('C = ', C)
        print('gamma = ', gamma)
        print('Accuracy:', scores)

# plot heatmap of MSE from grid search over eta and lambda
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_acc, annot=True, ax=ax, cmap="viridis", fmt = '.3g')
ax.set_yticklabels(C_vals)
ax.set_xticklabels(gamma_vals)
ax.set_title("Accuracy test data")
ax.set_xlabel("$\gamma$")
ax.set_ylabel("C")
fig.savefig('./Images/heatmap_svmclass1.png')
plt.show()

ngammas = 100
gammas = np.logspace(-4, 4, ngammas)
test_acc_g = np.zeros(ngammas)
i = 0
for gam in gammas:
    model = svm.SVC(kernel='rbf', gamma = gam)
    cv_ = cross_val_score(model, X_, y, cv = cv, n_jobs = -1, verbose = 0)
    test_acc_g[i] = np.mean(cv_)
    i += 1

nc = 100
cs = np.logspace(-4, 4, ngammas)
test_acc_c = np.zeros(ngammas)
i = 0
for c in cs:
    model = svm.SVC(kernel='rbf', C = c, gamma = 'auto')
    cv_ = cross_val_score(model, X_, y, cv = cv, n_jobs = -1, verbose = 0)
    test_acc_c[i] = np.mean(cv_)
    i += 1

fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True)
ax1.plot(np.log10(cs), test_acc_c, 'tab:orange')
ax1.set_title('C parameter')
ax1.set_ylabel('Accuracy score')
ax1.set_xlabel('log10(C)')
ax2.plot(np.log10(gammas), test_acc_g, 'tab:green')
ax2.set_title('$\gamma$ parameter')
ax2.set_xlabel('log10($\gamma$)')
fig.savefig('./Images/C_gamma_parameters')
plt.show()


# Optimized model
svclassifier = svm.SVC(kernel='rbf', C = 1, gamma = 1)
scores_SVMc = cross_val_score(svclassifier, X_, y, cv = cv)
#print(scores_SVMc)
print('CV accuracy score tuned SVMc: %0.3f (+/- %0.3f)' % (scores_SVMc.mean(), scores_SVMc.std()*2))


#### NEURAL NETWORK
print('-------------NN------------')

NNc_ = KerasClassifier(build_fn=NNc.build_network, verbose = 0)
scores_NNc = cross_val_score(NNc_, X_, y_onehot, cv = cv, verbose = 0)
#print(scores_NNc)
print('CV accuracy score NNc: %0.3f (+/- %0.3f)' % (scores_NNc.mean(), scores_NNc.std()*2))

# Find the number of epochs
history = NNc.build_network().fit(Xtrain, ytrain_onehot, validation_split=0.3, epochs=100, batch_size=32)
#print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("./Images/acc_hist_cat.png")
plt.show()

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./Images/loss_hist_cat.png')
#plt.show()


# Best regularization parameter
nlambdas = 15
lambdas = np.logspace(-8, 2, nlambdas)
test_acc = np.zeros(nlambdas)
i = 0
for lmbd in lambdas:
    DNN = NNclassifier(alpha=lmbd)
    DNN_ = KerasClassifier(build_fn=DNN.build_network)
    cv_DNN_ = cross_validate(DNN_, X_, y_onehot, cv = cv, return_train_score = False, n_jobs = -1)
    test_acc[i] = np.mean(cv_DNN_['test_score'])
    i += 1

plt.figure()
plt.plot(np.log10(lambdas), test_acc)
plt.xlabel('log10(lambda)')
plt.ylabel('Accuracy')
plt.title('NN classifier for different lambdas')
plt.savefig('Images/NNc_lambdas')
plt.show()

#grid search
NNc = NNclassifier(alpha = 10e-4)

parameters = {'layer_sizes':([128],[128,64],[256,128,64,32]), 'activation_function':['sigmoid','relu'], 'batch_size': [16,32,64]}
nnClassifier = KerasClassifier(build_fn=build_network, verbose=0)
gs_nn = GridSearchCV(nnClassifier, parameters, cv=cv, verbose=0, n_jobs=-1)
gs_nn.fit(Xtrain, ytrain_onehot)
best_parameters = gs_nn.best_params_
print('Best parameters GridSearch NN:', best_parameters)
#grid_predictions = grid.predict(Xtest)
#print(classification_report(ytest_onehot, grid_predictions))

NNc = NNclassifier(**gs_nn.best_params_, alpha = 10e-4)
NNc = KerasClassifier(build_fn=NNc.build_network, verbose=0)
scores_NNc = cross_val_score(NNc, X_, y_onehot, cv = cv)
print('CV accuracy score tuned NNc: %0.3f (+/- %0.3f)' % (scores_NNc.mean(), scores_NNc.std()*2))


#best params grid search: {'activation_function': 'relu', 'batch_size': 16, 'layer_sizes': [256, 128, 64, 32]}

# Optimized model
NNc_b_ = NNclassifier(activation_function = 'relu', batch_size = 16, layer_sizes = [256, 128, 64, 32], alpha = 10e-4)
NNc_b = KerasClassifier(build_fn=NNc_b_.build_network, verbose=0)
scores_NNc_b = cross_val_score(NNc_b, X_, y_onehot, cv = cv)
print('CV accuracy score tuned NNc: %0.3f (+/- %0.3f)' % (scores_NNc_b.mean(), scores_NNc_b.std()*2))

# Find the number of epochs
history = NNc_b_.build_network().fit(Xtrain, ytrain_onehot, validation_split=0.3, epochs=100, batch_size=32)
#print(history.history.keys())
fig1 = plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("./Images/acc_hist_cat_opt.png")
plt.show()

fig2 = plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./Images/loss_hist_cat_opt.png')

nlambdas = 15
lambdas = np.logspace(-8, 2, nlambdas)
test_acc = np.zeros(nlambdas)
i = 0
for lmbd in lambdas:
    DNN = NNclassifier(activation_function = 'relu', batch_size = 16, layer_sizes = [256, 128, 64, 32], alpha=lmbd)
    DNN_ = KerasClassifier(build_fn=DNN.build_network)
    cv_DNN_ = cross_validate(DNN_, X_, y_onehot, cv = cv, return_train_score = False, n_jobs = -1)
    test_acc[i] = np.mean(cv_DNN_['test_score'])
    i += 1

plt.figure()
plt.plot(np.log10(lambdas), test_acc)
plt.xlabel('log10(lambda)')
plt.ylabel('Accuracy')
plt.title('NN optimized classifier for different lambdas')
plt.savefig('Images/NNc_lambdas_opt')
plt.show()

"""
#######
# REGRESSION
#######

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 5)
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)
X_ = sc.transform(X)

#### LINEAR REGRESSION
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

OLS = LinearRegression()
Ridge = Ridge(alpha = 0.7)
LASSO = Lasso()

cv_OLS = cross_validate(OLS, X_, y, cv = cv, n_jobs = -1, scoring = scoring, return_train_score = False)
cv_Ridge = cross_validate(Ridge, X_, y, cv = cv, n_jobs = -1, scoring = scoring, return_train_score = False)
cv_LASSO = cross_validate(LASSO, X_, y, cv = cv, n_jobs = -1, scoring = scoring, return_train_score = False)

print('MSE OLS: %0.5f (+/- %0.5f)' % (-cv_OLS['test_MSE'].mean(), cv_OLS['test_MSE'].std()*2))
print('R2 OLS: %0.5f (+/- %0.5f)' % (cv_OLS['test_R2'].mean(), cv_OLS['test_R2'].std()*2))
print('MAD OLS: %0.5f (+/- %0.5f)' % (-cv_OLS['test_MAD'].mean(), cv_OLS['test_MAD'].std()*2))
print('---')

print('MSE Ridge: %0.5f (+/- %0.5f)' % (-cv_Ridge['test_MSE'].mean(), cv_Ridge['test_MSE'].std()*2))
print('R2 Ridge: %0.5f (+/- %0.5f)' % (cv_Ridge['test_R2'].mean(), cv_Ridge['test_R2'].std()*2))
print('MAD Ridge: %0.5f (+/- %0.5f)' % (-cv_Ridge['test_MAD'].mean(), cv_Ridge['test_MAD'].std()*2))
print('---')

print('MSE LASSO: %0.5f (+/- %0.5f)' % (-cv_LASSO['test_MSE'].mean(), cv_LASSO['test_MSE'].std()*2))
print('R2 LASSO: %0.5f (+/- %0.5f)' % (cv_LASSO['test_R2'].mean(), cv_LASSO['test_R2'].std()*2))
print('MAD LASSO: %0.5f (+/- %0.5f)' % (-cv_LASSO['test_MAD'].mean(), cv_LASSO['test_MAD'].std()*2))
print('---')


#SVM REGRESSOR
SVMr = svm.SVR(kernel = 'rbf', gamma = 'auto')

cv_SVMr = cross_validate(SVMr, X_, y, cv = cv, n_jobs = -1, scoring = scoring, return_train_score = False)
#print('cv_scores_nn_test:', cv_SVMr['test_MSE'])
print('MSE SVM: %0.5f (+/- %0.5f)' % (-cv_SVMr['test_MSE'].mean(), cv_SVMr['test_MSE'].std()*2))
print('R2 SVM: %0.5f (+/- %0.5f)' % (cv_SVMr['test_R2'].mean(), cv_SVMr['test_R2'].std()*2))
print('MAD SVM: %0.5f (+/- %0.5f)' % (-cv_SVMr['test_MAD'].mean(), cv_SVMr['test_MAD'].std()*2))
print('---')


C_vals = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4] #[0.01, 0.1, 1, 10, 1e2, 1e4, 1e6]
gamma_vals = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4] #[10, 1, 0.1, 0.01, 0.0001, 1e-5, 1e-7]
scores = np.zeros((len(C_vals), len(gamma_vals)))
test_mse = np.zeros((len(C_vals), len(gamma_vals)))
train_mse = np.zeros((len(C_vals), len(gamma_vals)))
for i, gamma in enumerate(gamma_vals):
    for j, C in enumerate(C_vals):
        model = svm.SVR(kernel = 'rbf', C = C, gamma = gamma)
        cv_ = cross_validate(model, X_, y, cv = cv, scoring = 'neg_mean_squared_error', n_jobs = -1, verbose = 0, return_train_score = True)
        #print(cv_.keys())
        scores = np.mean(-cv_['test_score'])
        test_mse[i][j] = np.mean(-cv_['test_score'])
        train_mse[i][j] = np.mean(-cv_['train_score'])
        #print('C = ', C)
        #print('gamma = ', gamma)
        #print('MSE:', scores)

# plot heatmap of MSE from grid search over eta and lambda
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_mse, annot=True, ax=ax, cmap="viridis", fmt = '.3g')
ax.set_yticklabels(gamma_vals)
ax.set_xticklabels(C_vals)
ax.set_title("MSE test data")
ax.set_ylabel("$\gamma$")
ax.set_xlabel("C")
fig.savefig('./Images/heatmap_svmreg_test.png')
plt.show()

SVMr_tuned = svm.SVR(kernel = 'rbf', gamma = 0.1, C = 1.0)
cv_SVMr_tuned = cross_validate(SVMr_tuned, X_, y, cv = cv, n_jobs = -1, scoring = scoring, return_train_score = False)
#print('cv_scores_nn_test:', cv_SVMr_tuned['test_MSE'])
print('MSE SVM_tuned: %0.5f (+/- %0.5f)' % (-cv_SVMr_tuned['test_MSE'].mean(), cv_SVMr_tuned['test_MSE'].std()*2))
print('R2 SVM_tuned: %0.5f (+/- %0.5f)' % (cv_SVMr_tuned['test_R2'].mean(), cv_SVMr_tuned['test_R2'].std()*2))
print('MAD SVM_tuned: %0.5f (+/- %0.5f)' % (-cv_SVMr_tuned['test_MAD'].mean(), cv_SVMr_tuned['test_MAD'].std()*2))
print('---')


##### NEURAL NETWORK REGRESSOR
layer_size = (100,20)
epochs = 50
batch_size = 16
lambda_ = 0
activation_function = 'relu'

NNr_ = NNregressor(layer_sizes = layer_size, activation_function = activation_function, alpha = lambda_, len_X = Xtrain)

# Fit the model
#history = NNr.build_network().fit(Xtrain, ytrain, validation_split=0.3, epochs=100, batch_size=32)

NNr = KerasRegressor(build_fn=NNr_.build_network, epochs = epochs, batch_size = batch_size, verbose = 0)
scores_NNr = cross_validate(NNr, X, y, cv = cv, scoring = scoring, return_train_score = False, verbose = 0, n_jobs = -1)
print('cv_scores_keras:', scores_NNr['test_MSE'])
print('MSE keras test: %0.5f (+/- %0.5f)' % (-np.mean(scores_NNr['test_MSE']), np.std(scores_NNr['test_MSE'])*2))
print('R2 keras test: %0.5f (+/- %0.5f)' % (np.mean(scores_NNr['test_R2']), np.std(scores_NNr['test_R2'])*2))
print('MAD keras test: %0.5f (+/- %0.5f)' % (-np.mean(scores_NNr['test_MAD']), np.std(scores_NNr['test_MAD'])*2))
print('---')

# Fit the model
history = NNr_.build_network().fit(Xtrain, ytrain, validation_split=0.3, epochs=100, batch_size=batch_size)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['mean_squared_error'], label='train')
plt.plot(history.history['val_mean_squared_error'], label='test')
plt.title('model MSE')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("./Images/mse_hist_reg.png")
plt.show()

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./Images/loss_hist_reg.png')
plt.show()

# Best regularization parameter
lmbd_vals = [0, 0.00001, 0.0001, 0.01, 0.1]
nlambdas = 10
lambdas = np.logspace(-7, 2, nlambdas)
test_MSE = np.zeros(nlambdas)
i = 0
for lmbd in lambdas:
    DNN = NNregressor(alpha=lmbd, len_X = Xtrain)
    DNN_ = KerasRegressor(build_fn=DNN.build_network)
    cv_DNN_ = cross_validate(DNN_, X, y, cv = cv, scoring = 'neg_mean_squared_error', return_train_score = False, n_jobs = -1)
    test_MSE[i] = np.mean(-cv_DNN_['test_score'])
    i += 1

plt.figure()
plt.plot(np.log10(lambdas), test_MSE)
plt.xlabel('log10(lambda)')
plt.ylabel('MSE')
plt.title('NN regression for different lambdas')
plt.show()

#grid search
parameters = {'layer_sizes':([128],[128,64],[256,128,64,32]), 'activation_function':['sigmoid', 'relu'], 'alpha':[0, 0.01, 0.03, 0.1], 'batch_size': [16,32,64]}
nnRegressor = KerasRegressor(build_fn=NNr_.build_network, verbose=0)
gs_nn = GridSearchCV(nnRegressor, parameters, scoring = 'mse', cv=cv, verbose = 0, n_jobs=-1)
gs_nn.fit(Xtrain, ytrain)
best_parameters = gs_nn.best_params_
print(best_parameters)

#NOT FINISHED

#optimized model
layer_size = (100,20)
epochs = 50
batch_size = 16
lambda_ = 0
activation_function = 'relu'

NNr_ = NNregressor(layer_sizes = layer_size, activation_function = activation_function, alpha = lambda_, len_X = Xtrain)

NNr = KerasRegressor(build_fn=NNr_.build_network, epochs = epochs, batch_size = batch_size, verbose = 0)
scores_NNr = cross_validate(NNr, X, y, cv = cv, scoring = scoring, return_train_score = False, verbose = 0, n_jobs = -1)
print('cv_scores_keras:', scores_NNr['test_MSE'])
print('MSE keras optimized: %0.5f (+/- %0.5f)' % (-np.mean(scores_NNr['test_MSE']), np.std(scores_NNr['test_MSE'])*2))
print('R2 keras optimized: %0.5f (+/- %0.5f)' % (np.mean(scores_NNr['test_R2']), np.std(scores_NNr['test_R2'])*2))
print('MAD keras optimized: %0.5f (+/- %0.5f)' % (-np.mean(scores_NNr['test_MAD']), np.std(scores_NNr['test_MAD'])*2))
print('---')
