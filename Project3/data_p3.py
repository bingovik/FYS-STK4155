import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

dfp = pd.read_csv('./Data/student-por.csv', sep = ';')

dfp[['Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
   'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']] = pd.Categorical(
   dfp[['Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
   'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']])

# Create design matrix and target variable
X = dfp.iloc[:, :30]
y = dfp.iloc[:, 30:]

#One hot encoding, it automatically does the cat/object variables
X_onehot = pd.get_dummies(X, drop_first=True)

sc = StandardScaler()


#######
#BINARY CLASSIFICATION
#######

y_b = y.copy()
y_b.values[y_b.values <= 9] = 0
y_b.values[y_b.values > 9] = 1

Xtrain, Xtest, ytrain, ytest = train_test_split(X_onehot, y_b, test_size = 0.2)
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)

ytrain_g3 = ytrain.iloc[:,2]
ytest_g3 = ytest.iloc[:,2]

logreg = LogisticRegression(solver = 'lbfgs').fit(Xtrain, ytrain_g3)
#pred = logreg.predict(Xtest)
#print(classification_report(ytest_g3, pred))

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
scores = cross_validate(logreg, Xtrain, ytrain_g3, cv = cv, scoring = 'accuracy')

print('CV accuracy score LOGREG: %0.3f (+/- %0.3f)' % (scores['test_score'].mean(), scores['test_score'].std()*2))
