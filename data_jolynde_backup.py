import pandas as pd
import os
import numpy as np
import random
import seaborn as sns
import pdb

from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate, ParameterGrid, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import matplotlib.patches

#import statsmodels.api as sm
from project2_functions import *
import classes_jolynde

#fignamePostFix = ''
#fignamePostFix = 'pca_remove_outliers'
fignamePostFix = '_remove_outliers'

# Read cedit card data to Pandas dataframe
df = pd.read_excel('./default of credit card clients.xls', header=1, skiprows = 0, index_col = 0)
df.rename(index=str, columns={"default payment next month": "defaultPayment"}, inplace=True)

print('Value count payment default:', df['defaultPayment'].value_counts())

# Drop some data that is likely unregistered
df = df.drop(df[(df.BILL_AMT1 == 0) &
                (df.BILL_AMT2 == 0) &
                (df.BILL_AMT3 == 0) &
                (df.BILL_AMT4 == 0) &
                (df.BILL_AMT5 == 0) &
                (df.BILL_AMT6 == 0)].index)

df = df.drop(df[(df.PAY_AMT1 == 0) &
                (df.PAY_AMT2 == 0) &
                (df.PAY_AMT3 == 0) &
                (df.PAY_AMT4 == 0) &
                (df.PAY_AMT5 == 0) &
                (df.PAY_AMT6 == 0)].index)

df = df.drop(df[(df.EDUCATION == 0) |
                (df.EDUCATION == 5) |
                (df.EDUCATION == 6)].index)

df = df.drop(df[(df.MARRIAGE == 0)].index)

# Remove outliers
data_todrop1 = (df[(df['LIMIT_BAL'] > 800000)].index)
data_todrop2 = df.loc[:, 'BILL_AMT1':'BILL_AMT6'][((df.loc[:, 'BILL_AMT1':'BILL_AMT6'] < 0) |(df.loc[:, 'BILL_AMT1':'BILL_AMT6'] > 700000)).any(axis=1)]
data_todrop3 = df.loc[:, 'PAY_AMT1':'PAY_AMT6'][((df.loc[:, 'PAY_AMT1':'PAY_AMT6'] < 0) | (df.loc[:, 'PAY_AMT1':'PAY_AMT6'] > 400000)).any(axis=1)]

data_todrop1 = data_todrop1.to_numpy()
data_todrop2 = data_todrop2.index.values
data_todrop3 = data_todrop3.index.values
data_todrop = np.concatenate((data_todrop1, data_todrop2, data_todrop3), axis=0)
data_todrop = np.unique(data_todrop)

df.drop(data_todrop, inplace = True)

# Correlation matrix
correlation_matrix = df.corr().round(1)
fig6, ax = plt.subplots(figsize=(15,15))
sns.heatmap(data=correlation_matrix, annot=True)
fig6.savefig('./Images/corr_matrix.png')
plt.show()

# Create the independent and dependent variables
X = df.loc[:, df.columns != 'defaultPayment'].values
y = df.loc[:, df.columns == 'defaultPayment'].values

# Categorical variables to one-hots
onehotencoder = OneHotEncoder(categories="auto")
X = ColumnTransformer(
    [("", onehotencoder, [2, 3]),],
    remainder="passthrough"
).fit_transform(X)
Y_onehot = onehotencoder.fit_transform(y).toarray()

# Make sure it's all integers, no float - to save memory/computational time?
X.astype(int)
y.astype(int)

# Split and scale the data
seed = 0
Xtrain, Xtest, ytrain, ytest, Y_train_onehot, Y_test_onehot = train_test_split(X, y, Y_onehot, test_size=0.2, random_state = seed)

sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)

'''
# PCA
pca = PCA(.97)  #Aim to keep 97% of variance and let algorithm find the appropriate number of principal components-
pca.fit(Xtrain)
print(pca.n_components_)
Xtrain = pca.transform(Xtrain)
Xtest = pca.transform(Xtest)
print(Xtrain.shape)

i=0
percento_range = np.arange(0.1,1,0.01)
n_comp = np.zeros(len(percento_range))
for percento in percento_range:
    pca = PCA(percento)
    pca.fit(Xtrain)
    n_comp[i] = pca.n_components_
    i = i+1
plot_several(np.hstack((n_comp[:,None],n_comp[:,None])), 100*np.hstack((percento_range[:,None],percento_range[-1-2]*np.ones(len(percento_range))[:,None])), ['r-', 'g-'], ['Variance', '97% variance'], 'Number of principal components', 'Variance (%)', 'Number of principal components vs variance covered', savefig = True, figname = 'PCA_analysis')
'''

#Our NN single test
nnTest = classes_jolynde.NeuralNetwork(n_hidden_neurons=(50,20), activation_function='sigmoid', lmbd=0.1, epochs=20, batch_size=64, eta=0.005)
history = nnTest.fit(Xtrain, Y_train_onehot, X_test = Xtest, y_test = Y_test_onehot, plot_learning=True)

#Keras/TensorFlow single test
nn_keras = build_network(layer_sizes=[50,20], alpha=0.1, activation_function='relu')
nn_keras.fit(Xtrain, Y_train_onehot, validation_data = (Xtest, Y_test_onehot), epochs = 80)
plot_several(np.repeat(np.arange(len(nn_keras.history.history['loss']))[:,None]+1, 2, axis=1), np.hstack((np.asarray(nn_keras.history.history['loss'])[:,None],np.asarray(nn_keras.history.history['val_loss'])[:,None])), ['r-', 'b-'], ['Train', 'Test'], 'Epochs', 'Loss', 'Training loss', savefig = False, figname = '')

#GridSearchCV on Tensorflow/Keras neural network
parameters = {'layer_sizes':([10],[50],[50,20]), 'activation_function':['sigmoid', 'relu'], 'alpha':[0, 0.01, 0.03, 0.1], 'epochs':[10,30,60]}
#parameters = {'layer_sizes':([10],[20]), 'activation_function':['sigmoid', 'relu'], 'alpha':[0, 0.01], 'epochs':[1,2]}
nnClassifier = KerasClassifier(build_fn=build_network, n_outputs=1, output_activation = 'sigmoid', loss="binary_crossentropy",verbose=0)
clf = GridSearchCV(nnClassifier, parameters, scoring = 'accuracy', cv=5, verbose = 8, n_jobs=-1)
clf.fit(Xtrain,ytrain)
df_grid_nn_Keras = pd.DataFrame.from_dict(clf.cv_results_)

#order data into matrix
df_grid_nn_Keras, row_names_nn_Keras, col_names_nn_Keras = order_gridSearchCV_data(df_grid_nn_Keras, column_param = 'alpha')

#plot heatmap of results
heatmap(df_grid_nn_Keras, 'Neural network (Keras/TensorFlow) validation accuracy (CV)', '\u03BB', 'parameters', col_names_nn_Keras, row_names_nn_Keras, True, savefig = True, figname = 'CV_Keras'+str(parameters)+fignamePostFix+'.png')

#fit best Tensorflow/Keras nn model and print metrics
print(clf.best_params_)
nnKerasBest = KerasClassifier(build_fn=build_network, n_outputs=1, output_activation = 'sigmoid', loss="binary_crossentropy",verbose=0)
nnKerasBest.set_params(**clf.best_params_)
nnKerasBest.fit(Xtrain, ytrain)
print('Classification report for TensorFlow neural network:')
print(classification_report(ytest, nnKerasBest.model.predict_classes(Xtest)))
print('Accuracy: %g' % accuracy_score(ytest,nnKerasBest.model.predict_classes(Xtest)))
area_ratio_nn_Tensor = area_ratio(ytest, nnKerasBest.predict_proba(Xtest)[:,1], plot = True, title = 'Lift Keras/Tensorflow neural network', savefig=True, figname = 'Lift_Keras'+str(clf.best_params_)+fignamePostFix+'.png')
print('Area ratio: %g' % area_ratio_nn_Tensor)
print('Confusion matrix for TensorFlow neural network:')
conf = confusion_matrix(ytest, nnKerasBest.model.predict_classes(Xtest))
print(conf)

#GridSearchCV on our own neural network
parameters = {'n_hidden_neurons':((10,),(50,),(50,20)), 'activation_function':['sigmoid', 'relu'], 'lmbd':[0, 0.01, 0.03, 0.1, 0.3], 'epochs':[10,30,60]}
#parameters = {'n_hidden_neurons':((10,),(50,)), 'activation_function':['sigmoid', 'relu'], 'lmbd':[0.1, 0.3], 'epochs':[1,3]}
nn = classes_jolynde.NeuralNetwork(eta=0.005)
clf = GridSearchCV(nn, parameters, scoring = 'accuracy', cv=5, verbose = 6)
clf.fit(Xtrain,Y_train_onehot)
df_grid_nn = pd.DataFrame.from_dict(clf.cv_results_)

#order data into matrix
data_array_nn, row_names_nn, col_names_nn = order_gridSearchCV_data(df_grid_nn, column_param = 'lmbd')

#plot heatmap of results
heatmap(data_array_nn, 'Neural network validation accuracy (CV)', '\u03BB', 'parameters', col_names_nn, row_names_nn, True, savefig = True, figname = 'CV_NN'+str(parameters)+fignamePostFix+'.png')

#fit best nn model and print metrics
print(clf.best_params_)
nnBest = classes_jolynde.NeuralNetwork(**clf.best_params_, eta=0.005)
nnBest.fit(Xtrain, Y_train_onehot, Xtest, Y_test_onehot, plot_learning=True, savefig=True, filename='NN_learning'+str(clf.best_params_)+'.png')
print('Classification report for neural network:')
print(classification_report(Y_test_onehot, nnBest.predict(Xtest)))
print('Accuracy: %g' % accuracy_score(Y_test_onehot,nnBest.predict(Xtest)))
area_ratio_nn = area_ratio(Y_test_onehot[:,1], nnBest.predict_proba(Xtest)[:,1], plot = True, title = 'Lift Neural Network', savefig=True, figname = 'Lift_NN'+str(clf.best_params_)+fignamePostFix+'.png')
print('Area ratio: %g' % area_ratio_nn)
print('Confusion matrix for neural network:')
conf = confusion_matrix(Y_test_onehot[:,1], nnBest.predict(Xtest)[:,1])
print(conf)

pdb.set_trace()

#GridSearchCV on our own logistic regression
parameters = {'_lambda':[0, 1, 3, 10, 30, 100], 'eta':[0.01,0.1,1, 3], 'max_iter':[100,500,1000]}
logReg = classes_jolynde.logisticRegression()
clf = GridSearchCV(logReg, parameters, scoring = 'accuracy', cv=5, verbose = 0)
clf.fit(Xtrain,ytrain)
df_grid_logReg = pd.DataFrame.from_dict(clf.cv_results_)

#order data into matrix
data_array_logReg, row_names_logReg, col_names_logReg = order_gridSearchCV_data(df_grid_logReg, column_param = '_lambda')

#plot heatmap of results
heatmap(data_array_logReg, 'Logistic regression validation accuracy (CV)', '\u03BB', 'parameters', col_names_logReg, row_names_logReg, True, savefig = True, figname = 'CV_logReg'+str(parameters)+fignamePostFix+'.png')

#fit best logistic regression model and print metrics
print(clf.best_params_)
logRegBest = classes_jolynde.logisticRegression(**clf.best_params_)
_, _ = logRegBest.fit_track_learning(Xtrain, ytrain, X_test = Xtest, y_test = ytest, plot = True, savefig = True, filename = 'logReg_learning')
print('Classification report for logistic regression:')
print(classification_report(ytest, logRegBest.predict(Xtest)))
print('Accuracy: %g' % accuracy_score(ytest,logRegBest.predict(Xtest)))
area_ratio_log = area_ratio(ytest, logRegBest.get_proba(Xtest), plot = True, title = 'Lift Logistic Regression', savefig=True, figname = 'Lift_logReg'+str(clf.best_params_)+fignamePostFix+'.png')
print('Area ratio: %g' % area_ratio_log)
print('Confusion matrix for logistic regression:')
conf = confusion_matrix(ytest, logRegBest.predict(Xtest))
print(conf)

pdb.set_trace()

#GridSearchCV on scikit-learn's logistic regression
parameters = {'C':[0.01, 0.1, 1, 10, 100], 'max_iter':[100,500,1000]}
logRegScikit = LogisticRegression()
clf = GridSearchCV(logRegScikit, parameters, scoring = 'accuracy', cv=5, verbose = 0)
clf.fit(Xtrain,ytrain)
df_grid_logRegScikit = pd.DataFrame.from_dict(clf.cv_results_)

#order data into matrix
data_array_logRegScikit, row_names_logRegScikit, col_names_logRegScikit = order_gridSearchCV_data(df_grid_logRegScikit, column_param = 'C')

#plot heatmap of results
heatmap(data_array_logRegScikit, 'Logistic regression (Scikit-Learn) validation accuracy (CV)', '1/\u03BB', 'parameters', col_names_logRegScikit, row_names_logRegScikit, True, savefig = True, figname = 'CV_logReg_Scikit'+str(parameters)+fignamePostFix+'.png')

#fit best Scikit learn logistic regression model and print metrics
print(clf.best_params_)
logRegScikitBest = LogisticRegression(**clf.best_params_)
logRegScikitBest.fit(Xtrain, ytrain)
print('Classification report for Scikit-learn logistic regression:')
print(classification_report(ytest, logRegScikitBest.predict(Xtest)))
print('Accuracy: %g' % accuracy_score(ytest,logRegScikitBest.predict(Xtest)))
area_ratio_log_scikit = area_ratio(ytest, logRegScikitBest.predict_proba(Xtest)[:,1], plot = True, title = 'Lift Logistic Regression (Scikit-Learn)', savefig=True, figname = 'Lift_logReg_Scikit'+str(clf.best_params_)+fignamePostFix+'.png')
print('Area ratio: %g' % area_ratio_log_scikit)
print('Confusion matrix for Scikit-learn logistic regression:')
conf = confusion_matrix(ytest, logRegScikitBest.predict(Xtest))
print(conf)
