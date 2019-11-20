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

from classes_p3 import *

dfr = pd.read_csv('./Data/winequality-red.csv', sep = ';')
dfw = pd.read_csv('./Data/winequality-red.csv', sep = ';')

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

from sklearn import svm
svclassifier = svm.SVC(kernel='rbf', gamma = 'auto')
#svclassifier.fit(Xtrain, ytrain)
#y_pred = svclassifier.predict(Xtest)

#print(confusion_matrix(ytest,y_pred))
#print(classification_report(ytest,y_pred))

scores_SVMc = cross_val_score(svclassifier, X_, y, cv = cv)
print(scores_SVMc)
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
ax.set_title("MSE test data")
ax.set_xlabel("$\gamma$")
ax.set_ylabel("C")
fig.savefig('./Images/heatmap_svmclass1.png')
plt.show()

ngammas = 100
gammas = np.logspace(-4, 4, ngammas)
test_acc = np.zeros(ngammas)
i = 0
for gam in gammas:
    model = svm.SVC(kernel='rbf', gamma = gam)
    cv_ = cross_val_score(model, X_, y, cv = cv, n_jobs = -1, verbose = 0)
    test_acc[i] = np.mean(cv_)
    i += 1

plt.figure()
plt.plot(np.log10(gammas), test_acc)
plt.xlabel('log10(gammas)')
plt.ylabel('accuracy')
plt.title('SVM classification for different gammas')
plt.show()


svclassifier = svm.SVC(kernel='rbf', C = 0.1, gamma = 0.1)
scores_SVMc = cross_val_score(svclassifier, X_, y, cv = cv)
print(scores_SVMc)
print('CV accuracy score tuned SVMc: %0.3f (+/- %0.3f)' % (scores_SVMc.mean(), scores_SVMc.std()*2))


"""
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


#######
# REGRESSION
#######

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2)
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)
X_ = sc.transform(X)

NNr_ = NNregressor(len_X = Xtrain)

# Fit the model
#history = NNr.build_network().fit(Xtrain, ytrain, validation_split=0.3, epochs=100, batch_size=32)

NNr = KerasRegressor(build_fn=NNr_.build_network, verbose = 1)
scores_NNr = cross_validate(NNr, X, y, cv = cv, scoring = ('neg_mean_squared_error', 'r2'), return_train_score = False, verbose = 1, n_jobs = -1)
print('cv_scores_keras:', scores_NNr['test_neg_mean_squared_error'])
print('MSE keras test: %0.5f (+/- %0.5f)' % (-np.mean(scores_NNr['test_neg_mean_squared_error']), np.std(scores_NNr['test_neg_mean_squared_error'])*2))
print('R2 keras test: %0.5f (+/- %0.5f)' % (np.mean(scores_NNr['test_r2']), np.std(scores_NNr['test_r2'])*2))

# Fit the model
history = NNr_.build_network().fit(Xtrain, ytrain, validation_split=0.3, epochs=100, batch_size=32)
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
"""
