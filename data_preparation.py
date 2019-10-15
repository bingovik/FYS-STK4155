import pandas as pd
import os
import numpy as np
import random
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import matplotlib.patches

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

# Create the independent and dependent variables
X = df.loc[:, df.columns != 'defaultPayment'].values
y = df.loc[:, df.columns == 'defaultPayment'].values

# Categorical variables to one-hot's
onehotencoder = OneHotEncoder(categories="auto")
X = ColumnTransformer(
    [("", onehotencoder, [2, 3]),],
    remainder="passthrough"
).fit_transform(X)

# Split and scale the data
seed = 1
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state = seed)

sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)

#Y_train_onehot, Y_test_onehot = onehotencoder.fit_transform(ytrain), onehotencoder.fit_transform(ytest)
