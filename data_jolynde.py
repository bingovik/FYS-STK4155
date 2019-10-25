import pandas as pd
import os
import numpy as np
import random
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import matplotlib.patches

from sklearn.decomposition import PCA

#import statsmodels.api as sm
from project2_functions import *
import classes_jolynde

import pdb

# Read the data
df = pd.read_excel('./default of credit card clients.xls', header=1, skiprows = 0, index_col = 0)
df.rename(index=str, columns={"default payment next month": "defaultPayment"}, inplace=True)

#sns.countplot(x = df['defaultPayment'], data = df)
#plt.title('Countplot default payment')
#plt.show()
#plt.savefig('./Images/Countplot_payment_default.png')

print('Value count payment default', df['defaultPayment'].value_counts())


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
"""
# Plot the variables
fig1, ax = plt.subplots(1,2)
sns.countplot(x = df['EDUCATION'], hue = df['defaultPayment'], data = df, ax = ax[0])
sns.countplot(x = df['MARRIAGE'], hue = df['defaultPayment'], data = df, ax = ax[1])
fig1.savefig('./Images/Education_marriage_plot.png')
#plt.show()

fig2, ax = plt.subplots(figsize=(10,5), ncols=3, nrows=2)
sns.countplot(x = df['PAY_0'], hue = df['defaultPayment'], data = df, ax = ax[0][0])
sns.countplot(x = df['PAY_2'], hue = df['defaultPayment'], data = df, ax = ax[0][1])
sns.countplot(x = df['PAY_3'], hue = df['defaultPayment'], data = df, ax = ax[0][2])
sns.countplot(x = df['PAY_4'], hue = df['defaultPayment'], data = df, ax = ax[1][0])
sns.countplot(x = df['PAY_5'], hue = df['defaultPayment'], data = df, ax = ax[1][1])
sns.countplot(x = df['PAY_6'], hue = df['defaultPayment'], data = df, ax = ax[1][2])
fig2.savefig('./Images/PAY_plots.png')
#plt.show()
"""

# Correlation matrix
correlation_matrix = df.loc[:, df.columns != 'defaultPayment'].corr().round(1)
sns.heatmap(data=correlation_matrix, annot=True)
plt.savefig('./Images/corr_matrix.png')
#plt.show()


# Create the independent and dependent variables
X = df.loc[:, df.columns != 'defaultPayment'].values
y = df.loc[:, df.columns == 'defaultPayment'].values


# Categorical variables to one-hot's
onehotencoder = OneHotEncoder(categories="auto")
X = ColumnTransformer(
    [("", onehotencoder, [2, 3]),],
    remainder="passthrough"
).fit_transform(X)

# Make sure its all integers, no float
X.astype(int)
y.astype(int)

# Split and scale the data
seed = 1
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state = seed)

sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)

# PCA
pca = PCA(.95)  #.95 for the number of components parameter. It means that scikit-learn choose the minimum number of principal components such that 95% of the variance is retained.
pca.fit(Xtrain)

print(pca.n_components_)
#print(pca.components_)

Xtrain_pca = pca.transform(Xtrain)
Xtest_pca = pca.transform(Xtest)
print(Xtrain_pca.shape)

Y_train_onehot, Y_test_onehot = onehotencoder.fit_transform(ytrain), onehotencoder.fit_transform(ytest)
Y_train_onehot = Y_train_onehot.toarray()
Y_test_onehot = Y_test_onehot.toarray()


nn = classes_jolynde.NeuralNetwork(n_hidden_neurons = (50,20), activation_function = 'relu')
nn.fit(Xtrain,Y_train_onehot, X_test = Xtest, y_test = Y_test_onehot, eta = 0.01, epochs = 40)
pdb.set_trace()
print(accuracy_score(ytest,nn.predict(Xtest)))

#GridSearchCV on our own neural network
parameters = {'n_hidden_neurons':((50,),(50,20)), 'activation_function':['sigmoid', 'relu'], 'lmbd':[0, 1]}
nn = classes_jolynde.NeuralNetwork()
clf = GridSearchCV(nn, parameters, scoring = 'accuracy', cv=5, verbose = 10)
clf.fit(Xtrain,Y_train_onehot, eta = 0.01, epochs = 40)
pdb.set_trace()

#clf = classes_jolynde.logReg_scikit()
#### LOGISTIC REGRESSION
clf = classes_jolynde.logReg_scikit()
clf_pca = classes_jolynde.logReg_scikit()
#clf = classes_jolynde.logisticRegression()
#clf_pca = classes_jolynde.logisticRegression()

# accuracy score
clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)
pdb.set_trace()
print("Accuracy score:", accuracy_score(ytest, ypred))

clf_pca.fit(Xtrain_pca, ytrain)
ypred_pca = clf_pca.predict(Xtest_pca)
print("Accuracy score PCA:", accuracy_score(ytest, ypred_pca))
"""
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
lift_chart(ytest, pred_prob, plot = True)

"""
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
