import pandas as pd
import os
import numpy as np
import random
import seaborn as sns
import pdb

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import matplotlib.patches

#import statsmodels.api as sm
from project2_functions import *
import classes_jolynde

# Read the data
df = pd.read_excel('./default of credit card clients.xls', header=1, skiprows = 0, index_col = 0)
df.rename(index=str, columns={"default payment next month": "defaultPayment"}, inplace=True)

#sns.countplot(x = df['defaultPayment'], data = df)
#plt.title('Countplot default payment')
#plt.show()
#plt.savefig('./Images/Countplot_payment_default.png')

print('Value count payment default:', df['defaultPayment'].value_counts())

# Drop some data that is unregistered
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


# Plot the variables
fig1, ax = plt.subplots(figsize = (14, 6), ncols = 3)
sns.countplot(x = df['EDUCATION'], hue = df['defaultPayment'], data = df, ax = ax[0])
sns.countplot(x = df['MARRIAGE'], hue = df['defaultPayment'], data = df, ax = ax[1])
sns.countplot(x = df['SEX'], hue = df['defaultPayment'], data = df, ax = ax[2])
fig1.savefig('./Images/Education_marriage_sex_plot.png')
#plt.show()

fig2, ax = plt.subplots(figsize=(10,8), ncols=3, nrows=2)
sns.countplot(x = df['PAY_0'], hue = df['defaultPayment'], data = df, ax = ax[0][0])
sns.countplot(x = df['PAY_2'], hue = df['defaultPayment'], data = df, ax = ax[0][1])
sns.countplot(x = df['PAY_3'], hue = df['defaultPayment'], data = df, ax = ax[0][2])
sns.countplot(x = df['PAY_4'], hue = df['defaultPayment'], data = df, ax = ax[1][0])
sns.countplot(x = df['PAY_5'], hue = df['defaultPayment'], data = df, ax = ax[1][1])
sns.countplot(x = df['PAY_6'], hue = df['defaultPayment'], data = df, ax = ax[1][2])
fig2.savefig('./Images/PAY_plots.png')
#plt.show()

# Remove outliers
df_out = df.copy()

def outlier_treatment(datacolumn):
    sorted(datacolumn)
    Q1 = datacolumn.quantile(0.25)
    Q3 = datacolumn.quantile(0.75)
    IQR = Q3 - Q1
    return Q1,Q3,IQR

Q1_LIMIT_BAL, Q3_LIMIT_BAL, iqr_LIMIT_BAL = outlier_treatment(df_out['LIMIT_BAL'])
l_limit = Q1_LIMIT_BAL - (1.5 * iqr_LIMIT_BAL)
u_limit = Q3_LIMIT_BAL + (1.5 * iqr_LIMIT_BAL)
data_todrop1 = (df_out[ (df_out['LIMIT_BAL'] > u_limit + 225000) | (df_out['LIMIT_BAL'] < l_limit) ].index)

Q1_BILL = df_out.loc[:, 'BILL_AMT1':'BILL_AMT6'].quantile(0.25)
Q3_BILL = df_out.loc[:, 'BILL_AMT1':'BILL_AMT6'].quantile(0.75)
IQR_BILL = Q3_BILL - Q1_BILL
data_todrop2 = df_out.loc[:, 'BILL_AMT1':'BILL_AMT6'][((df_out.loc[:, 'BILL_AMT1':'BILL_AMT6'] < (Q1_BILL - 1.5 * IQR_BILL)) |(df_out.loc[:, 'BILL_AMT1':'BILL_AMT6'] > (Q3_BILL + 1.5 * IQR_BILL) + 220000)).any(axis=1)]

Q1_PAY = df_out.loc[:, 'PAY_AMT1':'PAY_AMT6'].quantile(0.25)
Q3_PAY = df_out.loc[:, 'PAY_AMT1':'PAY_AMT6'].quantile(0.75)
IQR_PAY = Q3_PAY - Q1_PAY
data_todrop3 = df_out.loc[:, 'PAY_AMT1':'PAY_AMT6'][((df_out.loc[:, 'PAY_AMT1':'PAY_AMT6'] < (Q1_PAY - 1.5 * IQR_PAY)) |(df_out.loc[:, 'PAY_AMT1':'PAY_AMT6'] > (Q3_PAY + 1.5 * IQR_PAY) + 200000)).any(axis=1)]

data_todrop1 = data_todrop1.to_numpy()
data_todrop2 = data_todrop2.index.values
data_todrop3 = data_todrop3.index.values
data_todrop = np.concatenate((data_todrop1, data_todrop2, data_todrop3), axis=0)
data_todrop = np.unique(data_todrop)

df_out.drop(data_todrop, inplace = True)

"""
# Plot the variables with and without outliers to see the difference
fig3, ax = plt.subplots(figsize=(12,8), ncols=2, nrows=2)
sns.distplot(df['LIMIT_BAL'], ax = ax[0][0])
sns.distplot(df_out['LIMIT_BAL'], ax = ax[0][1])
sns.boxplot(x=df['LIMIT_BAL'], ax = ax[1][0])
sns.boxplot(x=df_out['LIMIT_BAL'], ax = ax[1][1])
ax[0, 0].set_title('LIMIT_BAL with outliers')
ax[0, 1].set_title('LIMIT_BAL without outliers')
fig3.savefig('./Images/LIMIT_BAL_plots.png')

fig4, ax = plt.subplots(figsize=(12,8), ncols=2, nrows=2)
sns.distplot(df['BILL_AMT1'], ax = ax[0][0])
sns.distplot(df_out['BILL_AMT1'], ax = ax[0][1])
sns.boxplot(x=df['BILL_AMT1'], ax = ax[1][0])
sns.boxplot(x=df_out['BILL_AMT1'], ax = ax[1][1])
ax[0, 0].set_title('BILL_AMT1 with outliers')
ax[0, 1].set_title('BILL_AMT1 without outliers')
fig4.savefig('./Images/BILL_AMT1_plots.png')

fig5, ax = plt.subplots(figsize=(12,8), ncols=2, nrows=2)
sns.distplot(df['PAY_AMT1'], ax = ax[0][0])
sns.distplot(df_out['PAY_AMT1'], ax = ax[0][1])
sns.boxplot(x=df['PAY_AMT1'], ax = ax[1][0])
sns.boxplot(x=df_out['PAY_AMT1'], ax = ax[1][1])
ax[0, 0].set_title('PAY_AMT1 with outliers')
ax[0, 1].set_title('PAY_AMT1 without outliers')
fig5.savefig('./Images/PAY_AMT1_plots.png')

g1 = sns.pairplot(df, hue = 'defaultPayment',
                 vars=['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
                )
g1.fig.suptitle("Pairplot BILL_AMT with outliers", y = 1.08)
g1.savefig('./Images/pairplot_BILL_AMT.png')

g2 = sns.pairplot(df_out, hue = 'defaultPayment',
                 vars=['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
                )
g2.fig.suptitle("Pairplot BILL_AMT without outliers", y = 1.08)
g2.savefig('./Images/pairplot_BILL_AMT_out.png')

g3 = sns.pairplot(df, hue = 'defaultPayment',
                 vars=['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
                )
g3.fig.suptitle("Pairplot PAY_AMT with outliers", y = 1.08)
g3.savefig('./Images/pairplot_PAY_AMT.png')

g4 = sns.pairplot(df_out, hue = 'defaultPayment',
                 vars=['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
                )
g4.fig.suptitle("Pairplot PAY_AMT without outliers", y = 1.08)
g4.savefig('./Images/pairplot_PAY_AMT_out.png')
"""

# Correlation matrix
#correlation_matrix = df.loc[:, df.columns != 'defaultPayment'].corr().round(1)
correlation_matrix = df.corr().round(1)
fig6, ax = plt.subplots(figsize=(15,15))
sns.heatmap(data=correlation_matrix, annot=True)
fig6.savefig('./Images/corr_matrix.png')
#plt.show()

# Create the independent and dependent variables
X = df.loc[:, df.columns != 'defaultPayment'].values
y = df.loc[:, df.columns == 'defaultPayment'].values

#X = df_out.loc[:, df.columns != 'defaultPayment'].values
#y = df_out.loc[:, df.columns == 'defaultPayment'].values

# Categorical variables to one-hot's
onehotencoder = OneHotEncoder(categories="auto")
X = ColumnTransformer(
    [("", onehotencoder, [2, 3]),],
    remainder="passthrough"
).fit_transform(X)

# Make sure it's all integers, no float - to save memory/computational time?
X.astype(int)
y.astype(int)

# Split and scale the data
seed = 0
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state = seed)

sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)

# PCA
pca = PCA(.95)  #.95 for the number of components parameter. It means that scikit-learn choose the minimum number of principal components such that 95% of the variance is retained.
pca.fit(Xtrain)
print(pca.n_components_)

Xtrain_pca = pca.transform(Xtrain)
Xtest_pca = pca.transform(Xtest)
print(Xtrain_pca.shape)

Y_train_onehot, Y_test_onehot = onehotencoder.fit_transform(ytrain), onehotencoder.fit_transform(ytest)
Y_train_onehot = Y_train_onehot.toarray()
Y_test_onehot = Y_test_onehot.toarray()

nn = classes_jolynde.NeuralNetwork(n_hidden_neurons = 50, activation_function = 'sigmoid', epochs = 40)
nn.fit(Xtrain,Y_train_onehot, X_test = Xtest, y_test = Y_test_onehot, plot_learning = False)
'''
#GridSearchCV on our own logistic regression
parameters = {'_lambda':[0, 0.1, 1, 10], 'eta':[0.01,0.1,1], 'max_iter':[100,500,1000]}
logReg = classes_jolynde.logisticRegression()
clf = GridSearchCV(logReg, parameters, scoring = 'accuracy', cv=5, verbose = 0)
clf.fit(Xtrain,ytrain)
df_grid_logReg = pd.DataFrame.from_dict(clf.cv_results_)

#fit best logistic regression model and print metrics
print(clf.best_params_)
logRegBest = classes_jolynde.logisticRegression(**clf.best_params_)
logRegBest.fit(Xtrain, ytrain)
print('Classification report for logistic regression:')
print(classification_report(ytest, logRegBest.predict(Xtest)))
print('Accuracy: %g' % accuracy_score(ytest,logRegBest.predict(Xtest)))
area_ratio_log = area_ratio(ytest, logRegBest.get_proba(Xtest))
print('Area ratio: %g' % area_ratio_log)
print('Confusion matrix for logistic regression:')
conf = confusion_matrix(ytest, logRegBest.predict(Xtest))
print(conf)

#GridSearchCV on scikit-learn's logistic regression
parameters = {'C':[0.1, 1, 10], 'max_iter':[100,500,1000]}
logRegScikit = LogisticRegression()
clf = GridSearchCV(logRegScikit, parameters, scoring = 'accuracy', cv=5, verbose = 0)
clf.fit(Xtrain,ytrain)
df_grid_logRegScikit = pd.DataFrame.from_dict(clf.cv_results_)

#fit best Scikit learn logistic regression model and print metrics
print(clf.best_params_)
logRegScikitBest = LogisticRegression(**clf.best_params_)
logRegScikitBest.fit(Xtrain, ytrain)
print('Classification report for Scikit-learn logistic regression:')
print(classification_report(ytest, logRegScikitBest.predict(Xtest)))
print('Accuracy: %g' % accuracy_score(ytest,logRegScikitBest.predict(Xtest)))
area_ratio_log_scikit = area_ratio(ytest, logRegScikitBest.predict_proba(Xtest)[:,1])
print('Area ratio: %g' % area_ratio_log_scikit)
print('Confusion matrix for Scikit-learn logistic regression:')
conf = confusion_matrix(ytest, logRegScikitBest.predict(Xtest))
print(conf)

#GridSearchCV on our own neural network
parameters = {'n_hidden_neurons':((10,),(50,),(50,20),(30,20,10)), 'activation_function':['sigmoid', 'relu'], 'lmbd':[0, 0.01, 0.03, 0.1], 'epochs':[10,40]}
nn = classes_jolynde.NeuralNetwork(eta=0.005)
clf = GridSearchCV(nn, parameters, scoring = 'accuracy', cv=5, verbose = 0)
clf.fit(Xtrain,Y_train_onehot)
df_grid_nn = pd.DataFrame.from_dict(clf.cv_results_)

#fit best nn model and print metrics
print(clf.best_params_)
nnBest = classes_jolynde.NeuralNetwork(**clf.best_params_, eta=0.005)
nnBest.fit(Xtrain, Y_train_onehot)
print('Classification report for neural network:')
print(classification_report(Y_test_onehot, nnBest.predict(Xtest)))
print('Accuracy: %g' % accuracy_score(Y_test_onehot,nnBest.predict(Xtest)))
area_ratio_nn = area_ratio(Y_test_onehot[:,1], nnBest.predict_proba(Xtest)[:,1])
print('Area ratio: %g' % area_ratio_nn)
print('Confusion matrix for neural network:')
conf = confusion_matrix(Y_test_onehot[:,1], nnBest.predict(Xtest)[:,1])
print(conf)
'''

nnTensor1 = classes_jolynde.Neural_TensorFlow()
nnTensor1.fit(Xtrain, ytrain)

pred = nnTensor1.predict_classes(Xtest)
print(pred.shape)
acc = accuracy_score(ytest, pred)
print(confusion_matrix(ytest, pred))

nn2 = classes_jolynde.NeuralTensorFlowGridSearch()
nn2.fit(Xtrain, ytrain)
pred2 = nn2.predict_classes
print(confusion_matrix(ytest, pred2))
'''
#GridSearchCV on Tensorflow/Keras neural network
parameters = {'layer_sizes':([50],[50,20],[32,16,8]), 'activation_function':['sigmoid', 'relu'], '_lambda':[0, 0.01, 0.1], 'epochs':[2]}
nnTensor = classes_jolynde.Neural_TensorFlow()
clf = GridSearchCV(nnTensor, parameters, scoring = 'accuracy', cv=2, verbose = 0)
clf.fit(Xtrain, ytrain)
df_grid_nn = pd.DataFrame.from_dict(clf.cv_results_)

print(clf.best_params_)

#fit best Tensorflow/Keras nn model and print metrics
nnBestTensor = classes_jolynde.Neural_TensorFlow(**clf.best_params_)
nnBestTensor.fit(Xtrain, ytrain)

print('Classification report for TensorFlow neural network:')
print(classification_report(ytest, nnBestTensor.predict_classes(Xtest)))
print('Accuracy: %g' % accuracy_score(ytest, nnBestTensor.predict_classes(Xtest)))
area_ratio_nn_Tensor = area_ratio(ytrain, nnBestTensor.predict(Xtest), plot = True)
print('Area ratio: %g' % area_ratio_nn_Tensor)
print('Confusion matrix for TensorFlow neural network:')
conf = confusion_matrix(ytest, nnBestTensor.predict_classes(Xtest))
print(conf)
'''
'''
#clf = classes_jolynde.logReg_scikit()

nn = classes_jolynde.NeuralNetwork(n_hidden_neurons = (50,20), activation_function = "relu")
nn.train(Xtrain,Y_train_onehot, eta = 0.01, epochs = 40)
#print('acc_normal:', accuracy_score(ytest, nn.predict(Xtest))

nn = classes_jolynde.NeuralNetwork(n_hidden_neurons = (50,20), activation_function = "relu")
nn.train(Xtrain_pca,Y_train_onehot, eta = 0.01, epochs = 40)
#print('acc_pca:', accuracy_score(ytest, nn.predict(Xtest_pca))


#### LOGISTIC REGRESSION

clf = classes_jolynde.logReg_scikit()
clf_pca = classes_jolynde.logReg_scikit()
#clf = classes_jolynde.logisticRegression()
#clf_pca = classes_jolynde.logisticRegression()

# accuracy score
clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)
print("Accuracy score:", accuracy_score(ytest, ypred))

clf_pca.fit(Xtrain_pca, ytrain)
ypred_pca = clf_pca.predict(Xtest_pca)
print("Accuracy score PCA:", accuracy_score(ytest, ypred_pca))

#scores = cross_val_score(clf.model, X, y.ravel(), cv = kfold)
#print("Accuracy_cv: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std()*2))

#print("Parameters:", clf.model.coef_)

##### NEURAL NETWORK
#nn = classes_jolynde.NeuralNetwork(Xtrain, Y_train_onehot, n_categories = 2)
#nn.train()

nn_sk = classes_jolynde.Neural_scikit()
nn_sk.fit(Xtrain, Y_train_onehot)
nn_sk_pred = nn_sk.predict_class(Xtest)
print(accuracy_score(ytest, nn_sk_pred))

conf = confusion_matrix(ytest, nn_sk_pred)
print(conf)

print(classification_report(ytest, nn_sk_pred))

pred_prob = nn_sk.predict(Xtest)[:, 1]
print(pred_prob)
#lift_chart(ytest, pred_prob, plot = True)

labels = ['Class 0', 'Class 1']

fig4 = plt.figure()
ax = fig4.add_subplot(111)
cax = ax.matshow(conf, cmap=plt.cm.Blues)
fig4.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
#plt.show()

TP = conf[1, 1]
TN = conf[0, 0]
FP = conf[0, 1]
FN = conf[1, 0]

print('Recall // Sensitivity:', (TP / float(FN + TP)))
print('Precision:', TP / float(TP + FP))

'''
#df_grid_df_grid_logReg = df_grid_df_grid_logReg.drop("mean_fit_time", axis = 1)
#df_grid_df_grid_logReg = df_grid_df_grid_logReg.drop("std_fit_time", axis = 1)
#df_grid_df_grid_logReg = df_grid_df_grid_logReg.drop("mean_score_time", axis = 1)
#df_grid_df_grid_logReg = df_grid_df_grid_logReg.drop("std_score_time", axis = 1)
