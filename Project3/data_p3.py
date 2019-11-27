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
#BINARY CLASSIFICATION
#######
"""
y_b = y.copy()
y_b[y_b <= 5] = 0
y_b[y_b > 5] = 1

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_b, test_size = 0.2)
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)
X_ = sc.transform(X)

logreg = LogisticRegression(solver = 'lbfgs').fit(Xtrain, ytrain)
#pred = logreg.predict(Xtest)
#print(classification_report(ytest_g3, pred))

scores_logreg = cross_val_score(logreg, X_, y_b, cv = cv, scoring = make_scorer(classification_report_with_accuracy_score))

print('Class_report LOGROG', classification_report(originalclass, predictedclass))
print('CV accuracy score LOGREG: %0.3f (+/- %0.3f)' % (scores_logreg.mean(), scores_logreg.std()*2))

def build_network(
            layer_sizes = (128,64,32,16),
            n_outputs = 1,
            batch_size = 64,
            epochs = 1000,
            loss = 'binary_crossentropy', #or accuracy
            alpha = 0,
            activation_function = 'relu',
            output_activation = 'sigmoid'
            ):
        model = Sequential()
        if isinstance(layer_sizes, int):
            layer_sizes = [layer_sizes]
        for layer_size in layer_sizes:
            model.add(Dense(layer_size, activation=activation_function, kernel_regularizer=regularizers.l2(alpha)))
        model.add(Dense(n_outputs, activation=output_activation))
        model.compile(loss=loss, optimizer='Adam', metrics=['accuracy'])
        return model

NN = KerasClassifier(build_fn=build_network)
scores_NN = cross_val_score(NN, X_, y_b, cv = cv)

print(scores_NN)

print('CV accuracy score NN: %0.3f (+/- %0.3f)' % (scores_NN.mean(), scores_NN.std()*2))


model = Sequential()
#model.add(Dense(128, activation='relu'))
#model.add(Dense(64, activation='relu'))
#model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(Xtrain, ytrain, validation_split=0.3, epochs=100, batch_size=16)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("./Images/acc_hist.png")
plt.show()

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./Images/loss_hist.png')
plt.show()
"""
"""
#######
#CATEGORICAL CLASSIFICATION
#######

onehotencoder = OneHotEncoder(categories = "auto")

#y_c = pd.get_dummies(y_c).values

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_c, test_size = 0.2)
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)
X_ = sc.transform(X)

NNc = NNclassifier()

svclassifier = svm.SVC(kernel='rbf', gamma = 'auto', C = 1.0) #default parameters
#svclassifier.fit(Xtrain, ytrain)
#y_pred = svclassifier.predict(Xtest)

#print(confusion_matrix(ytest,y_pred))
#print(classification_report(ytest,y_pred))

scores_SVMc = cross_val_score(svclassifier, X_, y, cv = cv)
#print(scores_SVMc)
print('CV accuracy score SVMc: %0.3f (+/- %0.3f)' % (scores_SVMc.mean(), scores_SVMc.std()*2))


C_vals = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4] #[0.01, 0.1, 1, 10, 1e2, 1e4, 1e6]
gamma_vals = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4] #[10, 1, 0.1, 0.01, 0.0001, 1e-5, 1e-7]
scores = np.zeros((len(C_vals), len(gamma_vals)))
test_acc = np.zeros((len(C_vals), len(gamma_vals)))
for i, gamma in enumerate(gamma_vals):
    for j, C in enumerate(C_vals):
        model = svm.SVC(kernel = 'rbf', C = C, gamma = gamma)
        cv_ = cross_val_score(model, X_, y, cv = cv, n_jobs = -1, verbose = 1)
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

svclassifier = svm.SVC(kernel='rbf', C = 0.1, gamma = 0.1)
scores_SVMc = cross_val_score(svclassifier, X_, y, cv = cv)
#print(scores_SVMc)
print('CV accuracy score tuned SVMc: %0.3f (+/- %0.3f)' % (scores_SVMc.mean(), scores_SVMc.std()*2))

# NEURAL NETWORK MOD
# Fit the model
history = NNc.build_network().fit(Xtrain, ytrain, validation_split=0.3, epochs=100, batch_size=32)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("./Images/acc_hist_cat.png")

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./Images/loss_hist_cat.png')
#plt.show()

#evaluate model
NNc = KerasClassifier(build_fn=NNc.build_network)
scores_NNc = cross_val_score(NNc, X_, y_c, cv = cv)

print(scores_NNc)

print('CV accuracy score NNc: %0.3f (+/- %0.3f)' % (scores_NNc.mean(), scores_NNc.std()*2))
"""

#######
# REGRESSION
#######

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2) #, random_state = 0)
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)
X_ = sc.transform(X)

#LINEAR REGRESSION
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

score = []
reg = LogisticRegression(solver = 'lbfgs', max_iter = 10000, multi_class = 'auto')

for i in range(100):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2) #, random_state = 0)
    reg.fit(Xtrain, ytrain)
    ypred = reg.predict(Xtest)
    print(i)
    score.append(accuracy_score(ytest, ypred))

sns.distplot(score, label = 'logreg')
plt.show()

"""
score = []
reg = Ridge(alpha = 0.7)

for i in range(1000):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2) #, random_state = 0)
    reg.fit(Xtrain, ytrain)
    ypred = reg.predict(Xtest)

    score.append(mean_squared_error(ytest, ypred))

sns.distplot(score, label = 'ridge')

score = []
reg1 = svm.SVR(kernel = 'rbf', gamma = 'auto')

for i in range(1000):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2) #, random_state = 0)
    reg1.fit(Xtrain, ytrain)
    ypred = reg1.predict(Xtest)

    score.append(mean_squared_error(ytest, ypred))

sns.distplot(score, label = 'svm')
plt.legend()
plt.show()

"""
"""
cv_reg = cross_validate(reg, X_, y, cv = cv, n_jobs = -1, scoring = scoring, return_train_score = False)

print('MSE OLS: %0.5f (+/- %0.5f)' % (-cv_reg['test_MSE'].mean(), cv_reg['test_MSE'].std()*2))
print('R2 OLS: %0.5f (+/- %0.5f)' % (cv_reg['test_R2'].mean(), cv_reg['test_R2'].std()*2))
print('MAD OLS: %0.5f (+/- %0.5f)' % (-cv_reg['test_MAD'].mean(), cv_reg['test_MAD'].std()*2))
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
        print(cv_.keys())
        scores = np.mean(-cv_['test_score'])
        test_mse[i][j] = np.mean(-cv_['test_score'])
        train_mse[i][j] = np.mean(-cv_['train_score'])
        print('C = ', C)
        print('gamma = ', gamma)
        print('MSE:', scores)

# plot heatmap of MSE from grid search over eta and lambda
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_mse, annot=True, ax=ax, cmap="viridis", fmt = '.3g')
ax.set_yticklabels(gamma_vals)
ax.set_xticklabels(C_vals)
ax.set_title("MSE train data")
ax.set_ylabel("$\gamma$")
ax.set_xlabel("C")
fig.savefig('./Images/heatmap_svmreg_train.png')
plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_mse, annot=True, ax=ax, cmap="viridis", fmt = '.3g')
ax.set_xticklabels(C_vals)
ax.set_yticklabels(gamma_vals)
ax.set_title("MSE test data")
ax.set_ylabel("$\gamma$")
ax.set_xlabel("C")
fig.savefig('./Images/heatmap_svmreg.png')
plt.show()


SVMr_tuned = svm.SVR(kernel = 'rbf', gamma = 0.1, C = 1.0)
cv_SVMr_tuned = cross_validate(SVMr_tuned, X_, y, cv = cv, n_jobs = -1, scoring = scoring, return_train_score = False)
#print('cv_scores_nn_test:', cv_SVMr_tuned['test_MSE'])
print('MSE SVM_tuned: %0.5f (+/- %0.5f)' % (-cv_SVMr_tuned['test_MSE'].mean(), cv_SVMr_tuned['test_MSE'].std()*2))
print('R2 SVM_tuned: %0.5f (+/- %0.5f)' % (cv_SVMr_tuned['test_R2'].mean(), cv_SVMr_tuned['test_R2'].std()*2))
print('MAD SVM_tuned: %0.5f (+/- %0.5f)' % (-cv_SVMr_tuned['test_MAD'].mean(), cv_SVMr_tuned['test_MAD'].std()*2))
print('---')

# NEURAL NETWORK REGRESSOR
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
"""
