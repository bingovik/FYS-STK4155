from project2_functions import *
import pandas as pd
import pdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, auc, roc_curve
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import scikitplot as skplt

class logisticRegression:

	def __init__(self, _lambda = 0, alpha = 0.1):
		self._lambda = _lambda; self.alpha = alpha;

	def train(self, X, y, max_iter = 100):
		_lambda = self._lambda; alpha = self.alpha
		self.beta = np.zeros(X.shape[1])[:,None]
		n = X.shape[0]
		for i in range(max_iter):
			y_pred = sigmoid(X@self.beta)
			grad = X.T@(y_pred - y)/n + np.vstack((0,_lambda*self.beta[1:]/n))
			self.beta = self.beta - alpha*grad

	def train_track_test(self, X, y, X_test, y_test, max_iter = 100, plot = False, savefig = False, filename = ''):
		_lambda = self._lambda; alpha = self.alpha
		self.beta = np.zeros(X.shape[1])[:,None]
		n = X.shape[0]
		cross_entropy_train = np.zeros(max_iter)
		cross_entropy_test = np.zeros(max_iter)
		for i in range(max_iter):
			y_pred = sigmoid(X@self.beta)
			y_pred_test = sigmoid(X_test@self.beta)
			cross_entropy_train[i] = categorical_cross_entropy(y_pred,y)
			cross_entropy_test[i] = categorical_cross_entropy(y_pred_test,y_test)
			#print(categorical_cross_entropy(y_pred,y),categorical_cross_entropy(y_pred_test,y_test))
			grad = X.T@(y_pred - y)/n + np.vstack((0,_lambda*self.beta[1:]/n))
			self.beta = self.beta - alpha*grad
		if plot:
			plot_several(np.repeat(np.arange(max_iter)[:,None], 2, axis=1),
				np.hstack((cross_entropy_train[:,None],cross_entropy_test[:,None])),
				['r-', 'b-'], ['train', 'test'],
				'iterations', 'cross entropy', 'Cross entropy during training',
				savefig = savefig, figname = filename)

	def predict(self, X):
		y_pred = sigmoid(X@self.beta)
		return y_pred

	def predict_outcome(self, X):
		y_pred_outcome = sigmoid(X@self.beta)
		y_pred_outcome[y_pred_outcome >= 0.5] = 1
		y_pred_outcome[y_pred_outcome < 0.5] = 0
		return y_pred_outcome

# Trying to set the seed
random.seed(0)

# Reading file into data frame
nanDict = {}
df = pd.read_excel('default of credit card clients.xls', header=1, skiprows=0, index_col=0, na_values=nanDict)
df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

# Remove instances with zeros only for past bill statements or paid amounts:
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

# Features and targets
X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values

# Categorical variables to one-hot's:
onehotencoder = OneHotEncoder(categories="auto")

#hard code Sex and Marrital status, should also take maybe education
X = ColumnTransformer(
    [("", onehotencoder, [1,3]),],
    remainder="passthrough"
).fit_transform(X)

#add bias column
X = np.hstack((np.ones(X.shape[0])[:,None], X))

#Train-test split
seed  = 1
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=seed)

#feature scaling, leaving bias column untouched
sc = StandardScaler()
X_train[:,1:] = sc.fit_transform(X_train[:,1:])
X_test[:,1:] = sc.transform(X_test[:,1:])

#setting hyper parameters
alpha = 0.1 #learning reate
_lambda = 160
lambda_tests = np.logspace(1, 3, num = 8)
alpha_tests = np.logspace(-1.5,0, num = 6)
max_iter = 1000
k = 10 #for k-folds cv


#creating and training logistic regression model
lg = logisticRegression(_lambda, alpha)
#lg.train(X_train, y_train, max_iter)
lg.train_track_test(X_train, y_train, X_test, y_test, max_iter, plot = True, savefig = False)

#predictions and metrics
y_test_pred = lg.predict(X_test)
y_test_pred_outcome = lg.predict_outcome(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred_outcome)
print(accuracy_test)
#calculate area ratio and plot lift chart
area_ratio = lift_chart(y_test,y_test_pred, plot = True, title = 'Lift chart: Logistic regression', savefig = False)

#precision-recall

#skplt.metrics.plot_confusion_matrix(y, predictions, normalize=True)

#cumulative gains/lift chart/area ratio

#cross validation
ce_val = np.zeros((len(lambda_tests),len(alpha_tests)))
ce_test = np.zeros((len(lambda_tests),len(alpha_tests)))
for i, lambda_test in enumerate(lambda_tests):
	for j, alpha_test in enumerate(alpha_tests):
		lg_cv = logisticRegression(lambda_test, alpha_test)
		ce_val[i,j], ce_test[i,j], _, _, _ = cv(lg_cv, k, categorical_cross_entropy, X_train, y_train, X_test, y_test, max_iter)
		print (ce_val[i,j], ce_test[i,j])
heatmap(ce_val, 'CE Validation', 'learning rate', '\u03BB', lambda_tests, alpha_tests, True, savefig = True, figname = 'Images/ce')
heatmap(ce_test, 'CE Test', 'learning rate', '\u03BB', lambda_tests, alpha_tests, True, savefig = True, figname = 'Images/ce')

pdb.set_trace()
slutt = 1
'''
# Change y to categorical using One-hot
y_train_cat, y_test_cat = onehotencoder.fit_transform(y_train), onehotencoder.fit_transform(y_test)
'''
