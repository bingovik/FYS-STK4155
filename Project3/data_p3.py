from warnings import simplefilter
simplefilter(action = 'ignore', category = FutureWarning)

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
from sklearn.utils import resample
from sklearn.metrics import make_scorer

from classes_p3 import *

dfr = pd.read_csv('./Data/winequality-red.csv', sep = ';')
dfw = pd.read_csv('./Data/winequality-white.csv', sep = ';')

# Create design matrix and target variable
X = dfw.iloc[:, :11].values
y = dfw.iloc[:, 11].values
y_c = dfw.iloc[:, 11]

# Holdout data for final test
X, X_T, y, y_T, y_c, y_c_T = train_test_split(X, y, y_c, test_size = 0.2, random_state = 2)

#One hot encoding, it automatically does the cat/object variables
sc = StandardScaler()
Xt = sc.fit_transform(X)
X_Tt = sc.transform(X_T)

originalclass = []
predictedclass = []
def classification_report_with_accuracy_score(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    return accuracy_score(y_true, y_pred) #return accuracy score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

def MAD(y_true, y_pred):
    return (sum(abs(y_true - y_pred)) / len(y_true))

scoring = {'MSE': 'neg_mean_squared_error', 'MAD': make_scorer(MAD, greater_is_better = False), 'R2': 'r2'}

#######
#CATEGORICAL CLASSIFICATION
#######

y_onehot = keras.utils.to_categorical(y_c, num_classes=None, dtype='float32')

Xtrain, Xtest, ytrain_onehot, ytest_onehot, ytrain_, ytest_ = train_test_split(X, y_onehot, y, test_size = 0.2, random_state = 2)
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)
X_ = sc.transform(X)

LR = LogisticRegression()
NNc = NNclassifier()
SVMc = svm.SVC(gamma = 'auto')

#### LOGISTIC REGRESSION
print('-------------LOGREG------------')
reg = LogisticRegression(solver = 'lbfgs', max_iter = 10000, multi_class = 'auto')
scores_logreg = cross_val_score(reg, X_, y, cv = cv)
print('CV accuracy score LogReg: %0.3f (+/- %0.3f)' % (scores_logreg.mean(), scores_logreg.std()*2))


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
fig, ax = plt.subplots(figsize = (8, 9))
sns.heatmap(test_acc, annot=True, ax=ax, cmap="viridis", fmt = '.3g')
ax.set_xticklabels(C_vals)
ax.set_yticklabels(gamma_vals)
ax.set_title("Accuracy test data")
ax.set_ylabel("$\gamma$")
ax.set_xlabel("C")
fig.savefig('./Images/heatmap_svmclass1.png')
plt.clf()
#plt.show()

# Find the C and gamma value of the maximum accuracy
result = np.where(test_acc == np.amax(test_acc))
listOfCordinates = list(zip(result[0], result[1]))
for cord in listOfCordinates:
    cord = cord
C_best = C_vals[cord[1]]
gamma_best = gamma_vals[cord[0]]
print('Optimal C value classification:', C_best)
print('Optimal gamma value classification:', gamma_best)

# Some more plots to show the distribution of the C and gammas
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

fig1, (ax1, ax2) = plt.subplots(1, 2, sharey = True)
ax1.plot(np.log10(cs), test_acc_c, 'tab:orange')
ax1.set_title('C parameter')
ax1.set_ylabel('Accuracy score')
ax1.set_xlabel('log10(C)')
ax2.plot(np.log10(gammas), test_acc_g, 'tab:green')
ax2.set_title('$\gamma$ parameter')
ax2.set_xlabel('log10($\gamma$)')
fig1.savefig('./Images/C_gamma_parameters')
plt.clf()
#plt.show()


# Optimized model
SVM_opt = svm.SVC(kernel='rbf', C = C_best, gamma = gamma_best)
scores_SVMc = cross_val_score(SVM_opt, X_, y, cv = cv)
#print(scores_SVMc)
print('CV accuracy score tuned SVMc: %0.3f (+/- %0.3f)' % (scores_SVMc.mean(), scores_SVMc.std()*2))


#### NEURAL NETWORK
print('-------------NN------------')

NNc_ = KerasClassifier(build_fn=NNc.build_network, verbose = 0)
scores_NNc = cross_val_score(NNc_, X_, y_onehot, cv = cv, verbose = 0)
#print(scores_NNc)
print('CV accuracy score NNc: %0.3f (+/- %0.3f)' % (scores_NNc.mean(), scores_NNc.std()*2))
"""
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
plt.clf()

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./Images/loss_hist_cat.png')
plt.clf()
"""
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

result = np.where(test_acc == np.amax(test_acc))
lambda_best = np.asscalar(lambdas[result[0]])
print('Optimal lambda value classification:', lambda_best)

plt.figure()
plt.plot(np.log10(lambdas), test_acc)
plt.xlabel('log10(lambda)')
plt.ylabel('Accuracy')
plt.title('NN classifier for different lambdas')
plt.savefig('Images/NNc_lambdas')
plt.clf()
#plt.show()


#Grid search over different parameters
NNc = NNclassifier(alpha = lambda_best)

parameters = {'layer_sizes':([128],[128,64],[256,128,64,32]), 'activation_function':['sigmoid','relu'], 'epochs': [20,50,100]}
nnClassifier = KerasClassifier(build_fn=build_network, alpha = lambda_best, verbose=0)
gs_nn = GridSearchCV(nnClassifier, parameters, cv=cv, verbose=0, n_jobs=-1)
gs_nn.fit(Xtrain, ytrain_onehot)
best_parameters = gs_nn.best_params_
print('Best parameters GridSearch NN:', best_parameters)
#grid_predictions = grid.predict(Xtest)
#print(classification_report(ytest_, grid_predictions))

NNc_b_ = NNclassifier(**gs_nn.best_params_, alpha = lambda_best)
NNc_b = KerasClassifier(build_fn=NNc_b_.build_network, verbose=0)
scores_NNc = cross_val_score(NNc_b, X_, y_onehot, cv = cv)
print('CV accuracy score tuned NNc: %0.3f (+/- %0.3f)' % (scores_NNc.mean(), scores_NNc.std()*2))


##### Final test on unseen data, bootstrapped to get error bars
print('-------------FINAL TEST------------')

pred_logreg = reg.fit(Xt, y).predict(X_Tt)
print('Accuracy LOGREG test:', accuracy_score(y_T, pred_logreg))
print(classification_report(y_T, pred_logreg))

pred_svm = SVM_opt.fit(Xt, y).predict(X_Tt)
print('Accuracy SVM test:', accuracy_score(y_T, pred_svm))
print(classification_report(y_T, pred_svm))

NNc_b_.fit(Xt, y_onehot)
pred_nn = NNc_b_.predict_classes(X_Tt)
print('Accuracy NN test:', accuracy_score(y_T, pred_nn))
print(classification_report(y_T, pred_nn))

models = list([reg, SVM_opt, NNc_b_])

quit()

n_bootstraps = 100
scores = np.empty((n_bootstraps, len(models)))

for m_int, m in enumerate(models):
    acc = []
    for i in range(n_bootstraps):
        x_, y_ = resample(X_Tt, y_T, n_samples = len(X_T))
        y_pred = m.predict(x_).ravel()
        acc.append(accuracy_score(y_, y_pred))

        scores[i, m_int] = accuracy_score(y_, y_pred)

    print(m, 'acc: %0.3f (+/- %0.3f)' % (np.mean(acc), np.std(acc)*2))
    sns.distplot(acc)

#Get some plots with the distribution of the accuracy
sns.distplot(scores[:,0], color="skyblue", label="Logistic Regression")
sns.distplot(scores[:,1], color="red", label="SVM")
sns.distplot(scores[:,2], color="green", label='Neural Network')
plt.legend()
plt.title('Accuracy scores on the test data')
plt.xlabel('Accuracy')
plt.savefig('Images/histplot_final_test (RS=2)')
plt.clf()
#plt.show()

fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6,8))
fig.suptitle('Accuracy score on the test data', fontsize = 15, y = 0.92)
sns.distplot(scores[:,0], ax=ax[0], color = 'skyblue', bins = 20)
ax[0].set_title('Logistic Regression', fontsize = 10, y = 0.85)
ax[0].xaxis.set_tick_params(which='both', labelbottom=True)

sns.distplot(scores[:,1], ax=ax[1], color = 'red', bins = 20)
ax[1].set_title('Support Vector Machine', fontsize = 10, y = 0.85)
ax[1].xaxis.set_tick_params(which='both', labelbottom=True)

sns.distplot(scores[:,2], ax=ax[2], color = 'green', bins = 20)
ax[2].set_title('Neural network', fontsize = 10, y = 0.85)
ax[2].set_xlabel('Accuracy', fontsize = 8)
plt.savefig('Images/histplots_seperate_final_test (RS=2)')
plt.clf()
#plt.show()



#######
# REGRESSION
#######

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 2)
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

print('MSE OLS: %0.3f (+/- %0.3f)' % (-cv_OLS['test_MSE'].mean(), cv_OLS['test_MSE'].std()*2))
print('R2 OLS: %0.3f (+/- %0.3f)' % (cv_OLS['test_R2'].mean(), cv_OLS['test_R2'].std()*2))
print('MAD OLS: %0.3f (+/- %0.3f)' % (-cv_OLS['test_MAD'].mean(), cv_OLS['test_MAD'].std()*2))
print('---')

print('MSE Ridge: %0.3f (+/- %0.3f)' % (-cv_Ridge['test_MSE'].mean(), cv_Ridge['test_MSE'].std()*2))
print('R2 Ridge: %0.3f (+/- %0.3f)' % (cv_Ridge['test_R2'].mean(), cv_Ridge['test_R2'].std()*2))
print('MAD Ridge: %0.3f (+/- %0.3f)' % (-cv_Ridge['test_MAD'].mean(), cv_Ridge['test_MAD'].std()*2))
print('---')

print('MSE LASSO: %0.3f (+/- %0.3f)' % (-cv_LASSO['test_MSE'].mean(), cv_LASSO['test_MSE'].std()*2))
print('R2 LASSO: %0.3f (+/- %0.3f)' % (cv_LASSO['test_R2'].mean(), cv_LASSO['test_R2'].std()*2))
print('MAD LASSO: %0.3f (+/- %0.3f)' % (-cv_LASSO['test_MAD'].mean(), cv_LASSO['test_MAD'].std()*2))
print('---')


#SVM REGRESSOR
SVMr = svm.SVR(kernel = 'rbf', gamma = 'auto')

cv_SVMr = cross_validate(SVMr, X_, y, cv = cv, n_jobs = -1, scoring = scoring, return_train_score = False)
#print('cv_scores_nn_test:', cv_SVMr['test_MSE'])
print('MSE SVM: %0.3f (+/- %0.3f)' % (-cv_SVMr['test_MSE'].mean(), cv_SVMr['test_MSE'].std()*2))
print('R2 SVM: %0.3f (+/- %0.3f)' % (cv_SVMr['test_R2'].mean(), cv_SVMr['test_R2'].std()*2))
print('MAD SVM: %0.3f (+/- %0.3f)' % (-cv_SVMr['test_MAD'].mean(), cv_SVMr['test_MAD'].std()*2))
print('---')


C_vals = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4] #[0.01, 0.1, 1, 10, 1e2, 1e4, 1e6]
gamma_vals = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4] #[10, 1, 0.1, 0.01, 0.0001, 1e-5, 1e-7]
scores = np.zeros((len(C_vals), len(gamma_vals)))
test_mse = np.zeros((len(C_vals), len(gamma_vals)))
#train_mse = np.zeros((len(C_vals), len(gamma_vals)))
for i, gamma in enumerate(gamma_vals):
    for j, C in enumerate(C_vals):
        model = svm.SVR(kernel = 'rbf', C = C, gamma = gamma)
        cv_ = cross_validate(model, X_, y, cv = cv, scoring = 'neg_mean_squared_error', n_jobs = -1, verbose = 0, return_train_score = True)
        scores = np.mean(-cv_['test_score'])
        test_mse[i][j] = np.mean(-cv_['test_score'])
        #train_mse[i][j] = np.mean(-cv_['train_score'])
        print('C = ', C)
        print('gamma = ', gamma)
        print('MSE:', scores)

# plot heatmap of MSE from grid search over eta and lambda
fig, ax = plt.subplots(figsize = (8, 9))
sns.heatmap(test_mse, annot=True, ax=ax, cmap="viridis", fmt = '.3g')
ax.set_yticklabels(gamma_vals)
ax.set_xticklabels(C_vals)
ax.set_title("MSE test data")
ax.set_ylabel("$\gamma$")
ax.set_xlabel("C")
fig.savefig('./Images/heatmap_svmreg_test.png')
plt.clf()
#plt.show()

# Find C and gamma with the minimum MSE value
result = np.where(test_mse == np.amin(test_mse))
listOfCordinates = list(zip(result[0], result[1]))
for cord in listOfCordinates:
    cord = cord
    print(cord)
C_best = C_vals[cord[1]]
gamma_best = gamma_vals[cord[0]]
print('Optimal C value regression:', C_best)
print('Optimal gamma value regression:', gamma_best)

#Optimized model
SVMr_opt = svm.SVR(kernel = 'rbf', gamma = gamma_best, C = C_best)
cv_SVMr_tuned = cross_validate(SVMr_opt, X_, y, cv = cv, n_jobs = -1, scoring = scoring, return_train_score = False)
#print('cv_scores_nn_test:', cv_SVMr_tuned['test_MSE'])
print('MSE SVM_tuned: %0.3f (+/- %0.3f)' % (-cv_SVMr_tuned['test_MSE'].mean(), cv_SVMr_tuned['test_MSE'].std()*2))
print('R2 SVM_tuned: %0.3f (+/- %0.3f)' % (cv_SVMr_tuned['test_R2'].mean(), cv_SVMr_tuned['test_R2'].std()*2))
print('MAD SVM_tuned: %0.3f (+/- %0.3f)' % (-cv_SVMr_tuned['test_MAD'].mean(), cv_SVMr_tuned['test_MAD'].std()*2))
print('---')


##### NEURAL NETWORK REGRESSOR
layer_size = (100,20)
epochs = 20
batch_size = 16
lambda_ = 0
activation_function = 'relu'

NNr_ = NNregressor(layer_sizes = layer_size, activation_function = activation_function, alpha = lambda_, len_X = Xtrain)

NNr = KerasRegressor(build_fn=NNr_.build_network, epochs = epochs, batch_size = batch_size, verbose = 0)
scores_NNr = cross_validate(NNr, X_, y, cv = cv, scoring = scoring, return_train_score = False, verbose = 0, n_jobs = -1)
print('cv_scores_keras:', scores_NNr['test_MSE'])
print('MSE NN: %0.3f (+/- %0.3f)' % (-np.mean(scores_NNr['test_MSE']), np.std(scores_NNr['test_MSE'])*2))
print('R2 NN: %0.3f (+/- %0.3f)' % (np.mean(scores_NNr['test_R2']), np.std(scores_NNr['test_R2'])*2))
print('MAD NN: %0.3f (+/- %0.3f)' % (-np.mean(scores_NNr['test_MAD']), np.std(scores_NNr['test_MAD'])*2))
print('---')

# Fit the model
history = NNr_.build_network().fit(Xtrain, ytrain, validation_split=0.3, epochs=100, batch_size=32)
# summarize history for accuracy
plt.figure(figsize = (6,4))
plt.plot(history.history['mean_squared_error'], label='train')
plt.plot(history.history['val_mean_squared_error'], label='test')
plt.title('model MSE')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("./Images/mse_hist_reg.png")
plt.clf()
#plt.show()

plt.figure(figsize = (6,4))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./Images/loss_hist_reg.png')
plt.clf()
#plt.show()

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

result = np.where(test_MSE == np.amin(test_MSE))
lambda_best = np.asscalar(lambdas[result[0]])
print('Optimal lambda value regression:', lambda_best)

plt.figure()
plt.plot(np.log10(lambdas), test_MSE)
plt.xlabel('log10(lambda)')
plt.ylabel('MSE')
plt.title('NN regression for different lambdas')
plt.savefig('Images/nn_reg_lambda')
plt.clf()
#plt.show()

# Grid search over different parameters, with the best alpha
parameters = {'layer_sizes':([128],[128,64],[256,128,64,32]), 'activation_function':['sigmoid', 'relu'], 'batch_size':[16,32,64]}

nnRegressor = KerasRegressor(build_fn=build_network, loss = 'mean_squared_error', output_activation = None, alpha = lambda_best, n_outputs = 1, verbose = 0)
gs_nn = GridSearchCV(nnRegressor, parameters, scoring = 'neg_mean_squared_error', cv=cv, verbose = 0, n_jobs=-1)
gs_nn.fit(Xtrain, ytrain)
print('Best parameters gridsearch NN:', gs_nn.best_params_)

# Optimized model
NNr_b_ = NNregressor(**gs_nn.best_params_, alpha = lambda_best, len_X = Xtrain)
NNr_b = KerasRegressor(build_fn=NNr_b_.build_network, epochs = epochs, batch_size = batch_size, verbose = 0)
scores_NNr_b = cross_validate(NNr_b, X_, y, cv = cv, scoring = scoring, return_train_score = False, verbose = 0, n_jobs = -1)
print('cv_scores_keras_opt:', scores_NNr_b['test_MSE'])
print('MSE tuned NN: %0.3f (+/- %0.3f)' % (-np.mean(scores_NNr_b['test_MSE']), np.std(scores_NNr_b['test_MSE'])*2))
print('R2 tuned NN: %0.3f (+/- %0.3f)' % (np.mean(scores_NNr_b['test_R2']), np.std(scores_NNr_b['test_R2'])*2))
print('MAD tuned NN: %0.3f (+/- %0.3f)' % (-np.mean(scores_NNr_b['test_MAD']), np.std(scores_NNr_b['test_MAD'])*2))
print('---')


##### Final test on unseen data, bootstrapped to get error bars
print('-------------FINAL TEST------------')

pred_OLS = OLS.fit(Xt, y).predict(X_Tt)
print('MSE OLS test:', mean_squared_error(y_T, pred_OLS))
print('Max prediction OLS:', pred_OLS.max())
print('Min prediction OLS:', pred_OLS.min())

pred_svm = SVMr_opt.fit(Xt, y).predict(X_Tt)
#pred_svm = svm.predict(X_T)
print('MSE SVM test:', mean_squared_error(y_T, pred_svm))
print('Max prediction SVM:', pred_svm.max())
print('Min prediction SVM:', pred_svm.min())

NNr_b_.fit(Xt, y)
pred_nnr = NNr_b_.predict(X_Tt)
print('MSE NN test:', mean_squared_error(y_T, pred_nnr))
print('Max prediction NN:', pred_nnr.max())
print('Min prediction NN:', pred_nnr.min())

models_reg = list([OLS, SVMr_opt, NNr_b_])

n_bootstraps = 100
scores_reg = np.empty((n_bootstraps, len(models_reg)))

for m_int, m in enumerate(models_reg):
    mse = []
    for i in range(n_bootstraps):
        x_, y_ = resample(X_Tt, y_T, n_samples = len(X_T))
        y_pred = m.predict(x_).ravel()
        mse.append(mean_squared_error(y_, y_pred))

        scores_reg[i, m_int] = mean_squared_error(y_, y_pred)

    print(m, 'MSE: %0.3f (+/- %0.3f)' % (np.mean(mse), np.std(mse)*2))

#Get some plots with the distribution of the MSE
sns.distplot(scores_reg[:,0], color="skyblue", label="Logistic Regression")
sns.distplot(scores_reg[:,1], color="red", label="SVM")
sns.distplot(scores_reg[:,2], color="green", label='Neural Network')
plt.title('MSE scores on the test data')
plt.xlabel('MSE')
plt.legend()
plt.savefig('Images/histplot_final_test_reg (RS=2)')
plt.clf()
#plt.show()

fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6,8))
fig.suptitle('MSE score on the test data', fontsize = 15, y = 0.92)
sns.distplot(scores_reg[:,0], ax=ax[0], color = 'skyblue', bins = 20)
ax[0].set_title('Logistic Regression', fontsize = 10, y = 0.85)
ax[0].xaxis.set_tick_params(which='both', labelbottom=True)

sns.distplot(scores_reg[:,1], ax=ax[1], color = 'red', bins = 20)
ax[1].set_title('Support Vector Machine', fontsize = 10, y = 0.85)
ax[1].xaxis.set_tick_params(which='both', labelbottom=True)

sns.distplot(scores_reg[:,2], ax=ax[2], color = 'green', bins = 20)
ax[2].set_title('Neural network', fontsize = 10, y = 0.85)
ax[2].set_xlabel('MSE', fontsize = 8)
plt.savefig('Images/histplots_seperate_final_test_reg (RS=2)')
plt.clf()
#plt.show()

scores_reg_mad = np.empty((n_bootstraps, len(models_reg)))

for m_int, m in enumerate(models_reg):
    mad = []
    for i in range(n_bootstraps):
        x_, y_ = resample(X_Tt, y_T, n_samples = len(X_T))
        y_pred = m.predict(x_).ravel()
        mad.append(MAD(y_, y_pred))

        scores_reg_mad[i, m_int] = MAD(y_, y_pred)

    print(m, 'MAD: %0.3f (+/- %0.3f)' % (np.mean(mad), np.std(mad)*2))

#Get some plots with the distribution of the MSE
sns.distplot(scores_reg_mad[:,0], color="skyblue", label="Logistic Regression")
sns.distplot(scores_reg_mad[:,1], color="red", label="SVM")
sns.distplot(scores_reg_mad[:,2], color="green", label='Neural Network')
plt.title('MAD scores on the test data')
plt.xlabel('MAD')
plt.legend()
plt.savefig('Images/histplot_final_test_reg_MAD (RS=2)')
plt.clf()
#plt.show()

fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6,8))
fig.suptitle('MAD score on the test data', fontsize = 15, y = 0.92)
sns.distplot(scores_reg_mad[:,0], ax=ax[0], color = 'skyblue', bins = 20)
ax[0].set_title('Logistic Regression', fontsize = 10, y = 0.85)
ax[0].xaxis.set_tick_params(which='both', labelbottom=True)

sns.distplot(scores_reg_mad[:,1], ax=ax[1], color = 'red', bins = 20)
ax[1].set_title('Support Vector Machine', fontsize = 10, y = 0.85)
ax[1].xaxis.set_tick_params(which='both', labelbottom=True)

sns.distplot(scores_reg_mad[:,2], ax=ax[2], color = 'green', bins = 20)
ax[2].set_title('Neural network', fontsize = 10, y = 0.85)
ax[2].set_xlabel('MAD', fontsize = 8)
plt.savefig('Images/histplots_seperate_final_test_reg_MAD (RS=2)')
plt.clf()
#plt.show()
