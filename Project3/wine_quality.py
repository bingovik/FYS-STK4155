from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import os
import numpy as np
import random
import seaborn as sns
import pdb
import pydot

from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical

from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate, ParameterGrid, cross_val_score, ShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from xgboost import XGBClassifier, XGBRegressor

import matplotlib.pyplot as plt
import matplotlib.patches

from project3_functions import *

wine_type = 'red' #red or white
savefigs = True
k = 5 #number of k-fold splits

#Read student performance data from cvs to Pandas dataframe
df = pd.read_csv('Data/winequality-' + wine_type + '.csv', sep=';',header=0)

#Histogram of numeric features
fig, ax = plt.subplots()
histplot = df.hist(ax = ax)
plt.savefig('Images/feature_hist_' + wine_type + '.png', dpi=300, bbox_inches='tight')
plt.show()

n_features = df.shape[1]

#Plot correlation matrix
heatmap(df.corr(), 'Feature correlation matrix, ' + wine_type, 'Features', 'Features', df.columns.tolist(), df.columns.tolist(), True, savefig = savefigs, figname = 'Images/corr_matrix_' + wine_type + '.png')

feature_list = list(df.columns[df.columns != 'quality'])

# Create the independent and dependent variables
X = df.loc[:, df.columns != 'quality'].values #including previous grades G1, G2
y = df.loc[:, df.columns == 'quality'].values

# Target variable to one-hots
onehotencoder = OneHotEncoder(categories="auto")
y_onehot = onehotencoder.fit_transform(y).toarray()
y_onehot = to_categorical(y)

# Split and scale the data
seed = 1
Xtrain, Xtest, ytrain, ytest, ytrain_onehot, ytest_onehot = train_test_split(X, y, y_onehot, test_size=0.2, random_state = seed)

sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)

#define score metrics for regression models
reg_scoring = {'MSE': 'neg_mean_squared_error', 'MAD': make_scorer(MAD), 'accuracy': make_scorer(accuracy_from_regression)}
'''
#NN classifier grid search
parameters = {'layer_sizes':([128,64],[256,128,64],[256,128,64,32]), 'alpha':[0, 0.00003, 0.0001, 0.0003, 0.001], 'epochs':[60,90,120,150]}
nnClassifier = KerasClassifier(build_fn=build_network, n_outputs=ytrain_onehot.shape[1], output_activation = 'softmax', loss="sparse_categorical_crossentropy",verbose=8)
clf = GridSearchCV(nnClassifier, parameters, scoring = 'accuracy', cv=5, verbose = 8, n_jobs=-1)
pdb.set_trace()
clf.fit(Xtrain, ytrain)
df_grid_nn_Keras_classifier = pd.DataFrame.from_dict(clf.cv_results_)

#order data into matrix
grid_nn_Keras_classifier, row_names_nn_Keras_classifier, col_names_nn_Keras_classifier = order_gridSearchCV_data(df_grid_nn_Keras_classifier, column_param = 'alpha')
print('Neural network classifier accuracy (CV): %g +-%g' % (df_grid_nn_Keras_classifier.mean_test_score.max(),2*np.mean(df_grid_nn_Keras_classifier.std_test_score[df_grid_nn_Keras_classifier.mean_test_score == df_grid_nn_Keras_classifier.mean_test_score.max()])))

#plot heatmap of results
heatmap(grid_nn_Keras_classifier, 'Neural network classifier accuracy (CV), '+ wine_type, '\u03BB', 'parameters', col_names_nn_Keras, row_names_nn_Keras, True, savefig = True, figname = 'Images/NN_clas_accuracy' + wine_type + '.png')

#refit best NN classifier
print(clf.best_params_)
nnKerasBest = KerasClassifier(build_fn=build_network, n_outputs=y_onehot.shape[1], output_activation = 'softmax', loss="categorical_crossentropy",verbose=0)
nnKerasBest.set_params(**clf.best_params_)
hist = nnKerasBest.fit(Xtrain, ytrain_onehot, validation_data=(Xtest,ytest_onehot))
print('Neural network classifier accuracy train: %g' % accuracy_score(ytrain,nnKerasBest.predict(Xtrain)))
print('Neural network classifier accuracy test: %g' % accuracy_score(ytest,nnKerasBest.predict(Xtest)))
'''
#XGBoost regressor grid search
parameters = {'eta': [0.2,0.3,0.4,1], 'gamma':[0,0.1,0.2,0.3,0.4,0.5], 'max_depth':[6,7,8,9,10,11]}
XGBoost_regressor = XGBRegressor(objective = 'reg:squarederror')
clf = GridSearchCV(XGBoost_regressor, parameters, scoring = reg_scoring, refit = 'MSE', cv=k, verbose = 0, n_jobs=-1)
clf.fit(Xtrain,ytrain)
df_grid_XGBoost_regressor = pd.DataFrame.from_dict(clf.cv_results_)

#order data into matrix
grid_XGBoost_regressor_MSE, row_names_XGBoost_regressor, col_names_XGBoost_regressor = order_gridSearchCV_data(df_grid_XGBoost_regressor, column_param = 'max_depth', score = 'mean_test_MSE')
grid_XGBoost_regressor_MAD, _, _ = order_gridSearchCV_data(df_grid_XGBoost_regressor, column_param = 'max_depth', score = 'mean_test_MAD')
grid_XGBoost_regressor_accuracy, _, _ = order_gridSearchCV_data(df_grid_XGBoost_regressor, column_param = 'max_depth', score = 'mean_test_accuracy')
print('XGBoost regressor CV validation MSE: %g +-%g' % (-df_grid_XGBoost_regressor.mean_test_MSE.max(),2*np.mean(df_grid_XGBoost_regressor.std_test_MSE[df_grid_XGBoost_regressor.mean_test_MSE == df_grid_XGBoost_regressor.mean_test_MSE.max()])))
print('XGBoost regressor CV validation MAD: %g +-%g' % (df_grid_XGBoost_regressor.mean_test_MAD.min(),2*np.mean(df_grid_XGBoost_regressor.std_test_MAD[df_grid_XGBoost_regressor.mean_test_MAD == df_grid_XGBoost_regressor.mean_test_MAD.min()])))
print('XGBoost regressor CV validation accuracy: %g +-%g' % (df_grid_XGBoost_regressor.mean_test_accuracy.max(),2*np.mean(df_grid_XGBoost_regressor.std_test_accuracy[df_grid_XGBoost_regressor.mean_test_accuracy == df_grid_XGBoost_regressor.mean_test_accuracy.max()])))

#plot heatmap of results
heatmap(-grid_XGBoost_regressor_MSE, 'XGboost regressor MSE (CV), '+ wine_type, 'max_depth', 'parameters', col_names_XGBoost_regressor, row_names_XGBoost_regressor, True, savefig = savefigs, figname = 'Images/XGBoost_reg_MSE_CV_' + wine_type + '.png')
heatmap(grid_XGBoost_regressor_MAD, 'XGboost regressor MAD (CV), '+ wine_type, 'max_depth', 'parameters', col_names_XGBoost_regressor, row_names_XGBoost_regressor, True, savefig = savefigs, figname = 'Images/XGBoost_reg_MAD' + wine_type + '.png')
heatmap(grid_XGBoost_regressor_accuracy, 'XGboost regressor accuracy (CV), '+ wine_type, 'max_depth', 'parameters', col_names_XGBoost_regressor, row_names_XGBoost_regressor, True, savefig = savefigs, figname = 'Images/XGBoost_reg_accuracy' + wine_type + '.png')

#Best XGboost regressor
XGBoost_regressor = XGBRegressor(**clf.best_params_, objective = 'reg:squarederror') #best with random seed for train/test spilt = 0
XGBoost_regressor.fit(Xtrain,ytrain)
print('XGboost regressor MSE train: %g' % mean_squared_error(ytrain,XGBoost_regressor.predict(Xtrain)))
print('XGboost regressor MSE test: %g' % mean_squared_error(ytest,XGBoost_regressor.predict(Xtest)))
print('XGboost regressor MAD train: %g' % MAD(ytrain,XGBoost_regressor.predict(Xtrain)))
print('XGboost regressor MAD test: %g' % MAD(ytest,XGBoost_regressor.predict(Xtest)))
print('XGboost regressor accuracy train: %g' % accuracy_score(ytrain,np.rint(XGBoost_regressor.predict(Xtrain))))
print('XGboost regressor accuracy test: %g' % accuracy_score(ytest,np.rint(XGBoost_regressor.predict(Xtest))))

#XGBoost classifier grid search
parameters = {'eta': [0.2,0.3,0.4,1], 'gamma':[0,0.1,0.2,0.3,0.4,0.5], 'max_depth':[6,7,8,9,10,11]}
XGBoost_classifier = XGBClassifier()
clf = GridSearchCV(XGBoost_classifier, parameters, scoring = 'accuracy', cv=k, verbose = 0, n_jobs=-1)
clf.fit(Xtrain,ytrain.ravel())
df_grid_XGBoost_classifier = pd.DataFrame.from_dict(clf.cv_results_)

#order data into matrix
grid_XGBoost_classifier, row_names_XGBoost_classifier, col_names_XGBoost_classifier = order_gridSearchCV_data(df_grid_XGBoost_classifier, column_param = 'max_depth')
print('XGBoost classifier CV validation accuracy: %g +-%g' % (df_grid_XGBoost_classifier.mean_test_score.max(),2*np.mean(df_grid_XGBoost_classifier.std_test_score[df_grid_XGBoost_classifier.mean_test_score == df_grid_XGBoost_classifier.mean_test_score.max()])))

#plot heatmap of results
heatmap(grid_XGBoost_classifier, 'XGboost classifier (CV), '+ wine_type, 'max_depth', 'parameters', col_names_XGBoost_classifier, row_names_XGBoost_classifier, True, savefig = savefigs, figname = 'Images/XGBoost_clas_accuracy_CV_' + wine_type + '.png')

#Best XGboost classifier
XGBoost_classifier = XGBClassifier(**clf.best_params_) #best with random seed for train/test spilt = 1
XGBoost_classifier.fit(Xtrain,ytrain.ravel())
print('XGboost classifier accuracy train: %g' % accuracy_score(ytrain,XGBoost_classifier.predict(Xtrain)))
print('XGboost classifier accuracy test: %g' % accuracy_score(ytest,XGBoost_classifier.predict(Xtest)))

#Ridge regression grid search
parameters = {'alpha':np.logspace(-1,2.5,15)}
Ridge_regression = Ridge()
clf = GridSearchCV(Ridge_regression, parameters, scoring = reg_scoring, refit = 'MSE', cv=k, verbose = 0, n_jobs=-1)
clf.fit(Xtrain,ytrain)
df_grid_Ridge_regression = pd.DataFrame.from_dict(clf.cv_results_)

#plot heatmap of results
heatmap(-df_grid_Ridge_regression['mean_test_MSE'].to_numpy()[:,None], 'Ridge MSE (CV), '+ wine_type, '', 'lambda', [1,2], np.logspace(-1,2.5,15)[:,None], True, savefig = savefigs, figname = 'Images/Ridge_reg_MSE_CV_' + wine_type + '.png')
heatmap(df_grid_Ridge_regression['mean_test_MAD'].to_numpy()[:,None], 'Ridge MAD (CV), '+ wine_type, '', 'lambda', [1,2], np.logspace(-1,2.5,15)[:,None], True, savefig = savefigs, figname = 'Images/Ridge_reg_MAD_CV_' + wine_type + '.png')
heatmap(df_grid_Ridge_regression['mean_test_accuracy'].to_numpy()[:,None], 'Ridge accuracy (CV), '+ wine_type, '', 'lambda', [1,2], np.logspace(-1,2.5,15)[:,None], True, savefig = savefigs, figname = 'Images/Ridge_reg_accuracy_CV_' + wine_type + '.png')
print('Ridge regression CV validation MSE: %g +-%g' % (-df_grid_Ridge_regression.mean_test_MSE.max(),2*np.mean(df_grid_Ridge_regression.std_test_MSE[df_grid_Ridge_regression.mean_test_MSE == df_grid_Ridge_regression.mean_test_MSE.max()])))
print('Ridge regression CV validation MAD: %g +-%g' % (df_grid_Ridge_regression.mean_test_MAD.min(),2*np.mean(df_grid_Ridge_regression.std_test_MAD[df_grid_Ridge_regression.mean_test_MAD == df_grid_Ridge_regression.mean_test_MAD.min()])))
print('Ridge regression CV validation accuracy: %g +-%g' % (df_grid_Ridge_regression.mean_test_accuracy.max(),2*np.mean(df_grid_Ridge_regression.std_test_accuracy[df_grid_Ridge_regression.mean_test_accuracy == df_grid_Ridge_regression.mean_test_accuracy.max()])))

#Best OLS regression (found no clear benefit from regularization in grid search)
OLS = LinearRegression()
OLS.fit(Xtrain,ytrain)
print('OLS MSE train: %g' % mean_squared_error(ytrain,OLS.predict(Xtrain)))
print('OLS MSE test: %g' % mean_squared_error(ytest,OLS.predict(Xtest)))
print('OLS MAD train: %g' % MAD(ytrain,OLS.predict(Xtrain)))
print('OLS MAD test: %g' % MAD(ytest,OLS.predict(Xtest)))
print('OLS accuracy train: %g' % accuracy_score(ytrain,np.rint(OLS.predict(Xtrain))))
print('OLS accuracy test: %g' % accuracy_score(ytest,np.rint(OLS.predict(Xtest))))

#intermezzo - estimate OLS model variance using bootstrap and cv
error, bias, variance = bootstrap_bias_variance_MSE(OLS, Xtrain, ytrain, 100, Xtest, ytest)
print('With bootstrap: OLS MSE=%g OLS bias:=%g OLS variance:=%g' % (error, bias, variance))
MSE_val, MSE_test, R2_val, bias_test_plus_noise, variance_test = cv(OLS, 5, mean_squared_error, Xtrain, ytrain, Xtest, ytest)

#Random forest regressor grid search
parameters = {'n_estimators':[100, 500, 1000], 'min_samples_leaf':[1,2,3], 'max_depth':(10, 11, 12, 13, None)}
rf_regressor = RandomForestRegressor(random_state=42)
clf = GridSearchCV(rf_regressor, parameters, scoring = reg_scoring, refit = 'MSE', cv=k, verbose = 0, n_jobs=-1)
clf.fit(Xtrain,ytrain.ravel())
df_grid_rf_regressor = pd.DataFrame.from_dict(clf.cv_results_)

#order data into matrix
grid_rf_regressor_MSE, row_names_rf_regressor, col_names_rf_regressor = order_gridSearchCV_data(df_grid_rf_regressor, column_param = 'min_samples_leaf', score = 'mean_test_MSE')
grid_rf_regressor_MAD, _, _ = order_gridSearchCV_data(df_grid_rf_regressor, column_param = 'min_samples_leaf', score = 'mean_test_MAD')
grid_rf_regressor_accuracy, _, _ = order_gridSearchCV_data(df_grid_rf_regressor, column_param = 'min_samples_leaf', score = 'mean_test_accuracy')
print('Random forest regression CV validation MSE: %g +-%g' % (-df_grid_rf_regressor.mean_test_MSE.max(),2*np.mean(df_grid_rf_regressor.std_test_MSE[df_grid_rf_regressor.mean_test_MSE == df_grid_rf_regressor.mean_test_MSE.max()])))
print('Random forest regression CV validation MAD: %g +-%g' % (df_grid_rf_regressor.mean_test_MAD.min(),2*np.mean(df_grid_rf_regressor.std_test_MAD[df_grid_rf_regressor.mean_test_MAD == df_grid_rf_regressor.mean_test_MAD.min()])))
print('Random forest regression CV validation accuracy: %g +-%g' % (df_grid_rf_regressor.mean_test_accuracy.max(),2*np.mean(df_grid_rf_regressor.std_test_accuracy[df_grid_rf_regressor.mean_test_accuracy == df_grid_rf_regressor.mean_test_accuracy.max()])))

#plot heatmap of results
heatmap(-grid_rf_regressor_MSE, 'Random forest MSE (CV), '+ wine_type, 'min_samples_leaf', 'parameters', col_names_rf_regressor, row_names_rf_regressor, True, savefig = savefigs, figname = 'Images/RF_reg_MSE_CV_' + wine_type + '.png')
heatmap(grid_rf_regressor_MAD, 'Random forest MAD (CV), '+ wine_type, 'min_samples_leaf', 'parameters', col_names_rf_regressor, row_names_rf_regressor, True, savefig = savefigs, figname = 'Images/RF_reg_MAD_CV_' + wine_type + '.png')
heatmap(grid_rf_regressor_accuracy, 'Random forest accuracy (CV), '+ wine_type, 'min_samples_leaf', 'parameters', col_names_rf_regressor, row_names_rf_regressor, True, savefig = savefigs, figname = 'Images/RF_reg_accuracy_CV_' + wine_type + '.png')

#Best random forest regressor
rf_regressor = RandomForestRegressor(**clf.best_params_, random_state=42)
rf_regressor.fit(Xtrain,ytrain.ravel())
print('Random forest regressor MSE train: %g' % mean_squared_error(ytrain,rf_regressor.predict(Xtrain)))
print('Random forest regressor MSE test: %g' % mean_squared_error(ytest,rf_regressor.predict(Xtest)))
print('Random forest regressor MAD train: %g' % MAD(ytrain,rf_regressor.predict(Xtrain)))
print('Random forest regressor MAD test: %g' % MAD(ytest,rf_regressor.predict(Xtest)))
print('Random forest regressor accuracy train: %g' % accuracy_score(ytrain,np.rint(rf_regressor.predict(Xtrain))))
print('Random forest regressor accuracy test: %g' % accuracy_score(ytest,np.rint(rf_regressor.predict(Xtest))))

#Random forest classifier grid search
parameters = {'n_estimators':[100, 500, 1000], 'min_samples_leaf':[1,2,3], 'max_depth':(10, 11, 12, 13, None)}
rf_classifier = RandomForestClassifier(random_state=42)
clf = GridSearchCV(rf_classifier, parameters, scoring = 'accuracy', cv=k, verbose = 0, n_jobs=-1)
clf.fit(Xtrain,ytrain.ravel())
df_grid_rf_classifier = pd.DataFrame.from_dict(clf.cv_results_)

#order data into matrix
grid_rf_classifier, row_names_rf_classifier, col_names_rf_classifier = order_gridSearchCV_data(df_grid_rf_classifier, column_param = 'min_samples_leaf')
print('Random forest classifier CV validation accuracy: %g +-%g' % (df_grid_rf_classifier.mean_test_score.max(),2*np.mean(df_grid_rf_classifier.std_test_score[df_grid_rf_classifier.mean_test_score == df_grid_rf_classifier.mean_test_score.max()])))

#plot heatmap of results
heatmap(grid_rf_classifier, 'Random forest (CV), ' + wine_type, 'min_samples_leaf', 'parameters', col_names_rf_classifier, row_names_rf_classifier, True, savefig = savefigs, figname = 'Images/RF_clas_CV_' + wine_type + '.png')

#Best random forest classifier
rf_classifier = RandomForestClassifier(**clf.best_params_, random_state=42)
rf_classifier.fit(Xtrain,ytrain.ravel())
print('Random forest classifier accuracy train: %g' % accuracy_score(ytrain, rf_classifier.predict(Xtrain)))
print('Random forest classifier accuracy test: %g' % accuracy_score(ytest, rf_classifier.predict(Xtest)))
print('Features: %s' % feature_list)
print('Random forest classifier feature importances: %s' % rf_classifier.feature_importances_)

#Simple decision tree for visualization
dt_classifier = DecisionTreeClassifier(max_depth=3)
dt_classifier.fit(sc.inverse_transform(Xtrain),ytrain.ravel())
print('Decision tree classifier accuracy train: %g' % accuracy_score(ytrain, dt_classifier.predict(sc.inverse_transform(Xtrain))))
print('Decision tree classifier accuracy test: %g' % accuracy_score(ytest, dt_classifier.predict(sc.inverse_transform(Xtest))))

# Export the image to a dot file
ytrain_labels = [str(m) for m in np.unique(ytrain)]
export_graphviz(dt_classifier, out_file = 'Images/DT_classifier_simple_' + wine_type + '.dot', feature_names = feature_list, rounded = True, precision = 1, filled = True, class_names = ytrain_labels)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('Images/DT_classifier_simple_' + wine_type + '.dot')
# Write graph to a png file
graph.write_png('Images/DT_classifier_simple_' + wine_type + '.dot')

#Decision tree classifier grid search
parameters = {'min_samples_leaf':[1,2,3], 'max_depth':(10, 11, 12, 13, None)}
dt_classifier = DecisionTreeClassifier(random_state=42)
clf = GridSearchCV(dt_classifier, parameters, scoring = 'accuracy', cv=k, verbose = 0, n_jobs=-1)
clf.fit(Xtrain,ytrain.ravel())
df_grid_dt_classifier = pd.DataFrame.from_dict(clf.cv_results_)

#order data into matrix
grid_dt_classifier, row_names_dt_classifier, col_names_dt_classifier = order_gridSearchCV_data(df_grid_dt_classifier, column_param = 'min_samples_leaf')
print('Decision tree classifier CV validation accuracy: %g +-%g' % (df_grid_dt_classifier.mean_test_score.max(),2*np.mean(df_grid_dt_classifier.std_test_score[df_grid_dt_classifier.mean_test_score == df_grid_dt_classifier.mean_test_score.max()])))

#plot heatmap of results
heatmap(grid_dt_classifier, 'Decision tree (CV)', 'min_samples_leaf', 'parameters', col_names_dt_classifier, row_names_dt_classifier, True, savefig = savefigs, figname = 'Images/DT_clas_accuracy_CV_' + wine_type + '.png')

#Best decision tree classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(Xtrain,ytrain.ravel())
print('Decision tree classifier accuracy train: %g' % accuracy_score(ytrain, dt_classifier.predict(Xtrain)))
print('Decision tree classifier accuracy test: %g' % accuracy_score(ytest, dt_classifier.predict(Xtest)))

pdb.set_trace()

#loop with different train test split and shuffling each time
rf_classifier = RandomForestClassifier(n_estimators = 1000)
accuracy_different_splits = np.zeros(10)
for i in range(10):
	Xtrain, Xtest, ytrain, ytest, ytrain_onehot, ytest_onehot = train_test_split(X, y, y_onehot, test_size=0.2)
	sc = StandardScaler()
	Xtrain = sc.fit_transform(Xtrain)
	Xtest = sc.transform(Xtest)
	rf_classifier.fit(Xtrain,ytrain.ravel())
	accuracy_different_splits[i] = accuracy_score(ytest,rf_classifier.predict(Xtest))
plt.hist(accuracy_different_splits)
plt.show()

#Show example tree
# Pull one tree from the forest
tree = rf_regressor.estimators_[5]
filtered_list = [feature for (feature, importance) in zip(feature_list, tree.feature_importances_>0) if importance]
# Export the image to a dot file
export_graphviz(tree, out_file = 'Images/RF_reg_tree_example_' + wine_type + '.dot', feature_names = feature_list, rounded = True, precision = 1, filled = True)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('Images/RF_reg_tree_example_' + wine_type + '.dot')
# Write graph to a png file
graph.write_png('Images/RF_reg_tree_example_' + wine_type + '.dot')
