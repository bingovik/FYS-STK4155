from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import os
import numpy as np
import random
import seaborn as sns
import pdb
#import pydot

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from keras.utils import to_categorical

from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate, ParameterGrid, cross_val_score, ShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, make_scorer, classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, export_graphviz

#from xgboost import XGBClassifier, XGBRegressor

import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.colors import ListedColormap

from project3_functions import *

wine_type = 'red' #red or white
savefigs = True
k = 5 #number of k-fold splits

#read wine data from cvs to Pandas dataframe
df = pd.read_csv('Data/winequality-' + wine_type + '.csv', sep=';',header=0)

#histogram of features
fig, ax = plt.subplots(figsize = (10,10))
histplot = df.hist(ax = ax)
plt.savefig('Images/feature_hist_' + wine_type + '.png', dpi=300, bbox_inches='tight')
#plt.show()
plt.clf()

n_features = df.shape[1]

#Plot correlation matrix
#heatmap(df.corr(), 'Feature correlation matrix, ' + wine_type, 'Features', 'Features', df.columns.tolist(), df.columns.tolist(), True, format = '.2f', savefig = savefigs, figname = 'Images/corr_matrix_' + wine_type + '.png')

correlation_matrix = df.corr().round(2)
fig6, ax = plt.subplots(figsize=(12,11))
sns.heatmap(data=correlation_matrix, annot=True, square = True)
fig6.suptitle('Feature correlation matrix,' + wine_type + 'dataset', y = 0.9, fontsize = 13)
fig6.savefig('./Images/corr_matrix_red' + wine_type + '.png')
#plt.show()
plt.clf()

quit()
feature_list = list(df.columns[df.columns != 'quality'])

# Create the independent and dependent variables
X = df.loc[:, df.columns != 'quality'].values #including previous grades G1, G2
y = df.loc[:, df.columns == 'quality'].values

# Target variable to categorical
onehotencoder = OneHotEncoder(categories="auto")
y_onehot = to_categorical(y)
#y_onehot = onehotencoder.fit_transform(y).toarray()

# Split and scale the data
seed = 1
Xtrain, Xtest, ytrain, ytest, ytrain_onehot, ytest_onehot = train_test_split(X, y, y_onehot, test_size=0.2, random_state = seed)

sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)

#define score metrics for regression models
reg_scoring = {'MSE': 'neg_mean_squared_error', 'MAD': make_scorer(MAD, greater_is_better = False), 'accuracy': make_scorer(accuracy_from_regression), 'R2': 'r2'}

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

############ Logistic regression ############
print('-------------Logistic regression------------')
reg = LogisticRegression(solver = 'lbfgs', max_iter = 10000, multi_class = 'auto')
scores_logreg = cross_val_score(reg, Xtrain, ytrain.ravel(), cv = cv)
print('CV accuracy score LogReg: %0.3f (+/- %0.3f)' % (scores_logreg.mean(), scores_logreg.std()*2))

#refit logistic regression and predicting on separate test set
pred_logreg = reg.fit(Xtrain, ytrain.ravel()).predict(Xtest)
print('Accuracy LOGREG test:', accuracy_score(ytest, pred_logreg))
print(classification_report(ytest, pred_logreg))

#confusion matrix
heatmap(confusion_matrix(ytest,pred_logreg),'Logistic regression confusion matrix, ' + wine_type, 'predicted', 'actual', np.unique(ytest), np.unique(ytest), True, format = '.0f', cmap = 'viridis', savefig = savefigs, figname = 'Images/LR_confusion_' + wine_type + '.png')

############ Random forest classifier ############
print('-------------Random forest classifier------------')

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
pred_rf_classifier_train = rf_classifier.predict(Xtrain)
pred_rf_classifier_test = rf_classifier.predict(Xtest)
print('Random forest classifier accuracy train: %g' % accuracy_score(ytrain, pred_rf_classifier_train))
print('Random forest classifier accuracy test: %g' % accuracy_score(ytest, pred_rf_classifier_test))
print('Features: %s' % feature_list)
print('Random forest classifier feature importances: %s' % rf_classifier.feature_importances_)

#confusion matrix
heatmap(confusion_matrix(ytest,pred_rf_classifier_test),'Random forest classifier confusion matrix, ' + wine_type, 'predicted', 'actual', np.unique(ytest), np.unique(ytest), True, format = '.0f', cmap = 'viridis', savefig = savefigs, figname = 'Images/RF_clas_confusion_' + wine_type + '.png')

############ Random forest regression ############
print('-------------Random forest regression------------')

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

#print cv results
print('Random forest regressor CV best validation MSE: %g +-%g' % (-df_grid_rf_regressor.mean_test_MSE.max(),2*np.mean(df_grid_rf_regressor.std_test_MSE[df_grid_rf_regressor.mean_test_MSE == df_grid_rf_regressor.mean_test_MSE.max()])))
print('Random forest regressor CV corresponding validation MAD: %g +-%g' % (-np.mean(df_grid_rf_regressor.mean_test_MAD[df_grid_rf_regressor.mean_test_MSE == df_grid_rf_regressor.mean_test_MSE.max()]),2*np.mean(df_grid_rf_regressor.std_test_MAD[df_grid_rf_regressor.mean_test_MSE == df_grid_rf_regressor.mean_test_MSE.max()])))
print('Random forest regressor CV corresponding validation accuracy: %g +-%g' % (np.mean(df_grid_rf_regressor.mean_test_accuracy[df_grid_rf_regressor.mean_test_MSE == df_grid_rf_regressor.mean_test_MSE.max()]),2*np.mean(df_grid_rf_regressor.std_test_accuracy[df_grid_rf_regressor.mean_test_MSE == df_grid_rf_regressor.mean_test_MSE.max()])))

#plot heatmap of results
heatmap(-grid_rf_regressor_MSE, 'Random forest MSE (CV), '+ wine_type, 'min_samples_leaf', 'parameters', col_names_rf_regressor, row_names_rf_regressor, True, savefig = savefigs, figname = 'Images/RF_reg_MSE_CV_' + wine_type + '.png')
heatmap(-grid_rf_regressor_MAD, 'Random forest MAD (CV), '+ wine_type, 'min_samples_leaf', 'parameters', col_names_rf_regressor, row_names_rf_regressor, True, savefig = savefigs, figname = 'Images/RF_reg_MAD_CV_' + wine_type + '.png')
heatmap(grid_rf_regressor_accuracy, 'Random forest accuracy (CV), '+ wine_type, 'min_samples_leaf', 'parameters', col_names_rf_regressor, row_names_rf_regressor, True, savefig = savefigs, figname = 'Images/RF_reg_accuracy_CV_' + wine_type + '.png')

#Best random forest regressor
rf_regressor = RandomForestRegressor(**clf.best_params_, random_state=42)
rf_regressor.fit(Xtrain,ytrain.ravel())
pred_rf_regressor_train = rf_regressor.predict(Xtrain)
pred_rf_regressor_test = rf_regressor.predict(Xtest)
print('Random forest regressor MSE train: %g' % mean_squared_error(ytrain,pred_rf_regressor_train))
print('Random forest regressor MSE test: %g' % mean_squared_error(ytest,pred_rf_regressor_test))
print('Random forest regressor MAD train: %g' % MAD(ytrain,pred_rf_regressor_train))
print('Random forest regressor MAD test: %g' % MAD(ytest,pred_rf_regressor_test))
print('Random forest regressor accuracy train: %g' % accuracy_score(ytrain,np.rint(pred_rf_regressor_train)))
print('Random forest regressor accuracy test: %g' % accuracy_score(ytest,np.rint(pred_rf_regressor_test)))

#confusion matrix
heatmap(confusion_matrix(ytest,np.rint(pred_rf_regressor_test)),'Random forest regression confusion matrix, ' + wine_type, 'predicted', 'actual', np.unique(ytest), np.unique(ytest), True, format = '.0f', cmap = 'viridis', savefig = savefigs, figname = 'Images/RF_reg_confusion_' + wine_type + '.png')

############ Neural network classifier ############
print('-------------Neural network classifier------------')

#NN classifier grid search
parameters = {'layer_sizes':([128,64],[256,128,64],[256,128,64,32]), 'alpha':[0, 0.00001, 0.00003, 0.0001, 0.0003], 'epochs':[30,60,90,120]}
nnClassifier = KerasClassifier(build_fn=build_network, n_outputs=ytrain_onehot.shape[1], output_activation = 'softmax', loss="sparse_categorical_crossentropy",verbose=0)
clf = GridSearchCV(nnClassifier, parameters, scoring = 'accuracy', cv=5, verbose = 0, n_jobs=-1)
clf.fit(Xtrain, ytrain)
df_grid_nn_Keras_classifier = pd.DataFrame.from_dict(clf.cv_results_)

#order data into matrix
grid_nn_Keras_classifier, row_names_nn_Keras_classifier, col_names_nn_Keras_classifier = order_gridSearchCV_data(df_grid_nn_Keras_classifier, column_param = 'alpha')

#print cv results
print('Neural network classifier accuracy (CV): %g +-%g' % (df_grid_nn_Keras_classifier.mean_test_score.max(),2*np.mean(df_grid_nn_Keras_classifier.std_test_score[df_grid_nn_Keras_classifier.mean_test_score == df_grid_nn_Keras_classifier.mean_test_score.max()])))

#plot heatmap of results
heatmap(grid_nn_Keras_classifier, 'Neural network classifier accuracy (CV), '+ wine_type, '\u03BB', 'parameters', col_names_nn_Keras_classifier, row_names_nn_Keras_classifier, True, savefig = True, figname = 'Images/NN_clas_accuracy_2' + wine_type + '.png')

#refit best NN classifier
print(clf.best_params_)
nnKerasBest = KerasClassifier(build_fn=build_network, n_outputs=y_onehot.shape[1], output_activation = 'softmax', loss="categorical_crossentropy",verbose=0)
nnKerasBest.set_params(**clf.best_params_)
hist = nnKerasBest.fit(Xtrain, ytrain_onehot, validation_data=(Xtest,ytest_onehot))
pred_nnKerasBest_train = nnKerasBest.predict(Xtrain)
pred_nnKerasBest_test = nnKerasBest.predict(Xtest)
print('Neural network classifier accuracy train: %g' % accuracy_score(ytrain,pred_nnKerasBest_train))
print('Neural network classifier accuracy test: %g' % accuracy_score(ytest,pred_nnKerasBest_test))

#learning chart for best model (accuracy and loss) and confusion matrix
plot_several(np.tile(np.arange(clf.best_params_['epochs'])[:,None],[1,2]), np.concatenate((np.reshape(hist.history['accuracy'],(clf.best_params_['epochs'],1)),np.reshape(hist.history['val_accuracy'],(clf.best_params_['epochs'],1))),axis=1), '', ['train','test'], 'epochs', 'accuracy', 'NN classifier learning, ' + wine_type, savefig = savefigs, figname = 'NN_clas_learning_acc' + wine_type + '.png')
plot_several(np.tile(np.arange(clf.best_params_['epochs'])[:,None],[1,2]), np.concatenate((np.reshape(hist.history['loss'],(clf.best_params_['epochs'],1)),np.reshape(hist.history['val_loss'],(clf.best_params_['epochs'],1))),axis=1), '', ['train','test'], 'epochs', 'loss', 'NN classifier learning, ' + wine_type, savefig = savefigs, figname = 'NN_clas_learning_loss' + wine_type + '.png')
heatmap(confusion_matrix(ytest,pred_nnKerasBest_test),'NN classifier confusion matrix, ' + wine_type, 'predicted', 'actual', np.unique(ytest), np.unique(ytest), True, format = '.0f', cmap = 'viridis', savefig = savefigs, figname = 'Images/NN_clas_confusion_' + wine_type + '.png')

############ Neural network regressor ############

print('-------------Neural network regressor------------')

#NN regressor grid search
parameters = {'layer_sizes':([128,64],[256,128,64],[256,128,64,32]), 'alpha':[0, 0.00001, 0.00003, 0.0001, 0.0003], 'epochs':[30,60,90,120]}
parameters = {'layer_sizes':([10],[50],[50,20],[128,64]), 'alpha':[0, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03], 'epochs':[5, 10, 15, 20, 30,60]}
nnRegressor = KerasRegressor(build_fn=build_network, n_outputs = 1, output_activation = None, loss="mean_squared_error",verbose=0)
clf = GridSearchCV(nnRegressor, parameters, scoring = reg_scoring, refit = 'MSE', cv=5, verbose = 0, n_jobs=-1)
clf.fit(Xtrain, ytrain)
df_grid_nn_Keras_regressor = pd.DataFrame.from_dict(clf.cv_results_)

#order data into matrix
grid_nn_Keras_regressor_MSE, row_names_nn_Keras_regressor, col_names_nn_Keras_regressor = order_gridSearchCV_data(df_grid_nn_Keras_regressor, column_param = 'alpha', score = 'mean_test_MSE')
grid_nn_Keras_regressor_MAD, _, _ = order_gridSearchCV_data(df_grid_nn_Keras_regressor, column_param = 'alpha', score = 'mean_test_MAD')
grid_nn_Keras_regressor_accuracy, _, _ = order_gridSearchCV_data(df_grid_nn_Keras_regressor, column_param = 'alpha', score = 'mean_test_accuracy')

#print cv results
print('Neural network regressor CV best validation MSE: %g +-%g' % (-df_grid_nn_Keras_regressor.mean_test_MSE.max(),2*np.mean(df_grid_nn_Keras_regressor.std_test_MSE[df_grid_nn_Keras_regressor.mean_test_MSE == df_grid_nn_Keras_regressor.mean_test_MSE.max()])))
print('Neural network regressor CV corresponding validation MAD: %g +-%g' % (-df_grid_nn_Keras_regressor.mean_test_MAD[df_grid_nn_Keras_regressor.mean_test_MSE == df_grid_nn_Keras_regressor.mean_test_MSE.max()],2*np.mean(df_grid_nn_Keras_regressor.std_test_MAD[df_grid_nn_Keras_regressor.mean_test_MSE == df_grid_nn_Keras_regressor.mean_test_MSE.max()])))
print('Neural network regressor CV corresponding validation accuracy: %g +-%g' % (df_grid_nn_Keras_regressor.mean_test_accuracy[df_grid_nn_Keras_regressor.mean_test_MSE == df_grid_nn_Keras_regressor.mean_test_MSE.max()],2*np.mean(df_grid_nn_Keras_regressor.std_test_accuracy[df_grid_nn_Keras_regressor.mean_test_MSE == df_grid_nn_Keras_regressor.mean_test_MSE.max()])))

#plot heatmap of results
heatmap(-grid_nn_Keras_regressor_MSE, 'Neural network regressor MSE (CV), '+ wine_type, '\u03BB', 'parameters', col_names_nn_Keras_regressor, row_names_nn_Keras_regressor, True, savefig = True, figname = 'Images/NN_reg_MSE_' + wine_type + '.png')
heatmap(-grid_nn_Keras_regressor_MAD, 'Neural network regressor MAD (CV), '+ wine_type, '\u03BB', 'parameters', col_names_nn_Keras_regressor, row_names_nn_Keras_regressor, True, savefig = True, figname = 'Images/NN_reg_MAD_' + wine_type + '.png')
heatmap(grid_nn_Keras_regressor_accuracy, 'Neural network regressor accuracy (CV), '+ wine_type, '\u03BB', 'parameters', col_names_nn_Keras_regressor, row_names_nn_Keras_regressor, True, savefig = True, figname = 'Images/NN_reg_accuracy_' + wine_type + '.png')

#refit best NN regressor
print(clf.best_params_)
nnKerasRegBest = KerasRegressor(build_fn=build_network, n_outputs=1, output_activation = None, loss="mean_squared_error",verbose=0)
nnKerasRegBest.set_params(**clf.best_params_)
hist = nnKerasRegBest.fit(Xtrain, ytrain, validation_data=(Xtest,ytest))
pred_nnKerasRegBest_train = nnKerasRegBest.predict(Xtrain)
pred_nnKerasRegBest_test = nnKerasRegBest.predict(Xtest)
print('Neural network regressor MSE train: %g' % mean_squared_error(ytrain,pred_nnKerasRegBest_train))
print('Neural network regressor MSE test: %g' % mean_squared_error(ytest,pred_nnKerasRegBest_test))
print('Neural network regressor MAD train: %g' % MAD(ytrain,pred_nnKerasRegBest_train))
print('Neural network regressor MAD test: %g' % MAD(ytest,pred_nnKerasRegBest_test))
print('Neural network regressor accuracy train: %g' % accuracy_score(ytrain,np.rint(pred_nnKerasRegBest_train)))
print('Neural network regressor accuracy test: %g' % accuracy_score(ytest,np.rint(pred_nnKerasRegBest_test)))

#learning chart for best model (accuracy and loss) and confusion matrix
plot_several(np.tile(np.arange(clf.best_params_['epochs'])[:,None],[1,2]), np.concatenate((np.reshape(hist.history['accuracy'],(clf.best_params_['epochs'],1)),np.reshape(hist.history['val_accuracy'],(clf.best_params_['epochs'],1))),axis=1), '', ['train','test'], 'epochs', 'accuracy', 'NN regressor learning, ' + wine_type, savefig = savefigs, figname = 'NN_reg_learning_acc' + wine_type + '.png')
plot_several(np.tile(np.arange(clf.best_params_['epochs'])[:,None],[1,2]), np.concatenate((np.reshape(hist.history['loss'],(clf.best_params_['epochs'],1)),np.reshape(hist.history['val_loss'],(clf.best_params_['epochs'],1))),axis=1), '', ['train','test'], 'epochs', 'loss', 'NN regressor learning, ' + wine_type, savefig = savefigs, figname = 'NN_reg_learning_loss' + wine_type + '.png')
heatmap(confusion_matrix(ytest,np.rint(pred_nnKerasRegBest_test)),'NN regressor confusion matrix, ' + wine_type, 'predicted', 'actual', np.unique(ytest), np.unique(ytest), True, format = '.0f', cmap = 'viridis', savefig = savefigs, figname = 'Images/NN_reg_confusion_' + wine_type + '.png')

############ XGBoost network regressor ############
print('-------------XGBoost regressor------------')

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

#print cv results
print('XGBoost regressor CV best validation MSE: %g +-%g' % (-df_grid_XGBoost_regressor.mean_test_MSE.max(),2*np.mean(df_grid_XGBoost_regressor.std_test_MSE[df_grid_XGBoost_regressor.mean_test_MSE == df_grid_XGBoost_regressor.mean_test_MSE.max()])))
print('XGBoost regressor CV corresponding validation MAD: %g +-%g' % (-np.mean(df_grid_XGBoost_regressor.mean_test_MAD[df_grid_XGBoost_regressor.mean_test_MSE == df_grid_XGBoost_regressor.mean_test_MSE.max()]),2*np.mean(df_grid_XGBoost_regressor.std_test_MAD[df_grid_XGBoost_regressor.mean_test_MSE == df_grid_XGBoost_regressor.mean_test_MSE.max()])))
print('XGBoost regressor CV corresponding validation accuracy: %g +-%g' % (np.mean(df_grid_XGBoost_regressor.mean_test_accuracy[df_grid_XGBoost_regressor.mean_test_MSE == df_grid_XGBoost_regressor.mean_test_MSE.max()]),2*np.mean(df_grid_XGBoost_regressor.std_test_accuracy[df_grid_XGBoost_regressor.mean_test_MSE == df_grid_XGBoost_regressor.mean_test_MSE.max()])))

#plot heatmap of results
heatmap(-grid_XGBoost_regressor_MSE, 'XGboost regressor MSE (CV), '+ wine_type, 'max_depth', 'parameters', col_names_XGBoost_regressor, row_names_XGBoost_regressor, True, savefig = savefigs, figname = 'Images/XGBoost_reg_MSE_CV_' + wine_type + '.png')
heatmap(-grid_XGBoost_regressor_MAD, 'XGboost regressor MAD (CV), '+ wine_type, 'max_depth', 'parameters', col_names_XGBoost_regressor, row_names_XGBoost_regressor, True, savefig = savefigs, figname = 'Images/XGBoost_reg_MAD' + wine_type + '.png')
heatmap(grid_XGBoost_regressor_accuracy, 'XGboost regressor accuracy (CV), '+ wine_type, 'max_depth', 'parameters', col_names_XGBoost_regressor, row_names_XGBoost_regressor, True, savefig = savefigs, figname = 'Images/XGBoost_reg_accuracy' + wine_type + '.png')

#Fit best XGboost regressor
XGBoost_regressor = XGBRegressor(**clf.best_params_, objective = 'reg:squarederror') #best with random seed for train/test spilt = 0
XGBoost_regressor.fit(Xtrain,ytrain)
pred_XGBoost_regressor_train = XGBoost_regressor.predict(Xtrain)
pred_XGBoost_regressor_test = XGBoost_regressor.predict(Xtest)
print('XGboost regressor MSE train: %g' % mean_squared_error(ytrain,pred_XGBoost_regressor_train))
print('XGboost regressor MSE test: %g' % mean_squared_error(ytest,pred_XGBoost_regressor_test))
print('XGboost regressor MAD train: %g' % MAD(ytrain,pred_XGBoost_regressor_train))
print('XGboost regressor MAD test: %g' % MAD(ytest,pred_XGBoost_regressor_test))
print('XGboost regressor accuracy train: %g' % accuracy_score(ytrain,np.rint(pred_XGBoost_regressor_train)))
print('XGboost regressor accuracy test: %g' % accuracy_score(ytest,np.rint(pred_XGBoost_regressor_test)))

#confusion matrix (with rounded predictions)
heatmap(confusion_matrix(ytest,np.rint(pred_XGBoost_regressor_test)),'XGBoost regressor confusion matrix, ' + wine_type, 'predicted', 'actual', np.unique(ytest), np.unique(ytest), True, format = '.0f', cmap = 'viridis', savefig = savefigs, figname = 'Images/XGBoost_reg_confusion_' + wine_type + '.png')

############ XGBoost classifier ############
print('-------------XGBoost classifier------------')

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
XGBoost_classifier = XGBClassifier(**clf.best_params_)
XGBoost_classifier.fit(Xtrain,ytrain.ravel())
pred_XGBoost_classifier_train = XGBoost_classifier.predict(Xtrain)
pred_XGBoost_classifier_test = XGBoost_classifier.predict(Xtest)
print('XGboost classifier accuracy train: %g' % accuracy_score(ytrain,pred_XGBoost_classifier_train))
print('XGboost classifier accuracy test: %g' % accuracy_score(ytest,pred_XGBoost_classifier_test))

#confusion matrix
heatmap(confusion_matrix(ytest,pred_XGBoost_classifier_test),'XGBoost classifier confusion matrix, ' + wine_type, 'predicted', 'actual', np.unique(ytest), np.unique(ytest), True, format = '.0f', cmap = 'viridis', savefig = savefigs, figname = 'Images/XGBoost_clas_confusion_' + wine_type + '.png')

############ Ridge regression ############
print('-------------Ridge regression------------')

#Ridge regression grid search
parameters = {'alpha':np.logspace(-1,2.5,15)}
Ridge_regression = Ridge()
clf = GridSearchCV(Ridge_regression, parameters, scoring = reg_scoring, refit = 'MSE', cv=k, verbose = 0, n_jobs=-1)
clf.fit(Xtrain,ytrain)
df_grid_Ridge_regression = pd.DataFrame.from_dict(clf.cv_results_)

#plot heatmap of results
heatmap(-df_grid_Ridge_regression['mean_test_MSE'].to_numpy()[:,None], 'Ridge MSE (CV), '+ wine_type, '', 'lambda', [1,2], np.logspace(-1,2.5,15)[:,None], True, savefig = savefigs, figname = 'Images/Ridge_reg_MSE_CV_' + wine_type + '.png')
heatmap(-df_grid_Ridge_regression['mean_test_MAD'].to_numpy()[:,None], 'Ridge MAD (CV), '+ wine_type, '', 'lambda', [1,2], np.logspace(-1,2.5,15)[:,None], True, savefig = savefigs, figname = 'Images/Ridge_reg_MAD_CV_' + wine_type + '.png')
heatmap(df_grid_Ridge_regression['mean_test_accuracy'].to_numpy()[:,None], 'Ridge accuracy (CV), '+ wine_type, '', 'lambda', [1,2], np.logspace(-1,2.5,15)[:,None], True, savefig = savefigs, figname = 'Images/Ridge_reg_accuracy_CV_' + wine_type + '.png')

#print cv results
print('Ridge regression CV best validation MSE: %g +-%g' % (-df_grid_Ridge_regression.mean_test_MSE.max(),2*np.mean(df_grid_Ridge_regression.std_test_MSE[df_grid_Ridge_regression.mean_test_MSE == df_grid_Ridge_regression.mean_test_MSE.max()])))
print('Ridge regression CV corresponding validation MAD: %g +-%g' % (-np.mean(df_grid_Ridge_regression.mean_test_MAD[df_grid_Ridge_regression.mean_test_MSE == df_grid_Ridge_regression.mean_test_MSE.max()]),2*np.mean(df_grid_Ridge_regression.std_test_MAD[df_grid_Ridge_regression.mean_test_MSE == df_grid_Ridge_regression.mean_test_MSE.max()])))
print('Ridge regression CV corresponding validation accuracy: %g +-%g' % (np.mean(df_grid_Ridge_regression.mean_test_accuracy[df_grid_Ridge_regression.mean_test_MSE == df_grid_Ridge_regression.mean_test_MSE.max()]),2*np.mean(df_grid_Ridge_regression.std_test_accuracy[df_grid_Ridge_regression.mean_test_MSE == df_grid_Ridge_regression.mean_test_MSE.max()])))

#Refitting best Ridge regression
RidgeBest = Ridge(**clf.best_params_)
RidgeBest.fit(Xtrain,ytrain)
pred_RidgeBest_train = RidgeBest.predict(Xtrain)
pred_RidgeBest_test = RidgeBest.predict(Xtest)
print('Ridge MSE train: %g' % mean_squared_error(ytrain,pred_RidgeBest_train))
print('Ridge MSE test: %g' % mean_squared_error(ytest,pred_RidgeBest_test))
print('Ridge MAD train: %g' % MAD(ytrain,pred_RidgeBest_train))
print('Ridge MAD test: %g' % MAD(ytest,pred_RidgeBest_test))
print('Ridge accuracy train: %g' % accuracy_score(ytrain,np.rint(pred_RidgeBest_train)))
print('Ridge accuracy test: %g' % accuracy_score(ytest,np.rint(pred_RidgeBest_test)))

#confusion matrix
heatmap(confusion_matrix(ytest,np.rint(pred_RidgeBest_test)),'Ridge regression confusion matrix, ' + wine_type, 'predicted', 'actual', np.unique(ytest), np.unique(ytest), True, format = '.0f', cmap = 'viridis', savefig = savefigs, figname = 'Images/Ridge_confusion_' + wine_type + '.png')

#intermezzo - estimate Ridge model variance using bootstrap and cv
error, bias, variance = bootstrap_bias_variance_MSE(RidgeBest, Xtrain, ytrain, 100, Xtest, ytest)
print('With bootstrap: Ridge MSE=%g Ridge bias:=%g Ridge variance:=%g' % (error, bias, variance))
MSE_val, MSE_test, R2_val, bias_test_plus_noise, variance_test = crossVal(RidgeBest, 5, mean_squared_error, Xtrain, ytrain, Xtest, ytest)

######### SUPPORT VECTOR MACHINE classifier #########
print('-------------SVM classifier------------')

# Find the best values for the hyperparameters of the SVM
C_vals = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4] #[0.01, 0.1, 1, 10, 1e2, 1e4, 1e6]
gamma_vals = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4] #[10, 1, 0.1, 0.01, 0.0001, 1e-5, 1e-7]
test_acc = np.zeros((len(C_vals), len(gamma_vals)))
cv_mean_max = -1
for i, gamma in enumerate(gamma_vals):
    for j, C in enumerate(C_vals):
        model = svm.SVC(kernel = 'rbf', C = C, gamma = gamma)
        cv_ = cross_val_score(model, Xtrain, ytrain.ravel(), cv = cv, n_jobs = -1, verbose = 0)
        cv_mean = np.mean(cv_)
        test_acc[i][j] = cv_mean
        if cv_mean > cv_mean_max:
        	cv_mean_max = cv_mean
        	cv_std_best = np.std(cv_)
        print('C = %g, gamma = %g, accuracy = %g' % (C,gamma,cv_mean))

print('CV best accuracy score tuned SVMc: %0.3f (+/- %0.3f)' % (cv_mean_max, cv_std_best*2))

# plot heatmap of MSE from grid search over eta and lambda
fig, ax = plt.subplots(figsize = (8, 9))
sns.heatmap(test_acc, annot=True, ax=ax, cmap="viridis", fmt = '.3g')
ax.set_xticklabels(C_vals)
ax.set_yticklabels(gamma_vals)
ax.set_title("Accuracy test data")
ax.set_ylabel("$\gamma$")
ax.set_xlabel("C")
fig.savefig('./Images/heatmap_svmclass_red.png')
plt.clf()
#plt.show()

# Find the C and gamma value giving the maximum accuracy
result = np.where(test_acc == np.amax(test_acc))
C_best = C_vals[result[1][0]]
gamma_best = gamma_vals[result[0][0]]
print('Optimal C value classification:', C_best)
print('Optimal gamma value classification:', gamma_best)

##### Final test on unseen data, bootstrapped to get error bars
SVM_opt = svm.SVC(kernel='rbf', C = C_best, gamma = gamma_best)
pred_svm = SVM_opt.fit(Xtrain, ytrain).predict(Xtest)
print('Accuracy SVM test:', accuracy_score(ytest, pred_svm))
print(classification_report(ytest, pred_svm))

#confusion matrix
heatmap(confusion_matrix(ytest,pred_svm),'SVM classifier confusion matrix, ' + wine_type, 'predicted', 'actual', np.unique(ytest), np.unique(ytest), True, format = '.0f', cmap = 'viridis', savefig = savefigs, figname = 'Images/SVM_clas_confusion_' + wine_type + '.png')
plt.clf()

############ SVM REGRESSOR ############
print('-------------Support vector machine regressor------------')

#initial run with default settings
SVMr = svm.SVR(kernel = 'rbf', gamma = 'auto')
cv_SVMr = cross_validate(SVMr, Xtrain, ytrain.ravel(), cv = cv, n_jobs = -1, scoring = reg_scoring, return_train_score = False)
print('SVM (default settings) MSE: %0.3f (+/- %0.3f)' % (-cv_SVMr['test_MSE'].mean(), cv_SVMr['test_MSE'].std()*2))
print('SVM (default settings) R2: %0.3f (+/- %0.3f)' % (cv_SVMr['test_R2'].mean(), cv_SVMr['test_R2'].std()*2))
print('SVM (default settings) MAD: %0.3f (+/- %0.3f)' % (-cv_SVMr['test_MAD'].mean(), cv_SVMr['test_MAD'].std()*2))
print('---')

#grid search to optimize C and gamma
C_vals = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4] #[0.01, 0.1, 1, 10, 1e2, 1e4, 1e6]
gamma_vals = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4] #[10, 1, 0.1, 0.01, 0.0001, 1e-5, 1e-7]
test_mse = np.zeros((len(C_vals), len(gamma_vals)))
for i, gamma in enumerate(gamma_vals):
    for j, C in enumerate(C_vals):
        model = svm.SVR(kernel = 'rbf', C = C, gamma = gamma)
        cv_ = cross_validate(model, Xtrain, ytrain.ravel(), cv = cv, scoring = 'neg_mean_squared_error', n_jobs = -1, verbose = 0, return_train_score = True)
        scores = np.mean(-cv_['test_score'])
        test_mse[i][j] = np.mean(-cv_['test_score'])
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
fig.savefig('./Images/heatmap_svmreg_test_red.png')
plt.clf()
#plt.show()

# Find the C and gamma value giving the minimum MSE
result = np.where(test_mse == np.amin(test_mse))
C_best = C_vals[result[1][0]]
gamma_best = gamma_vals[result[0][0]]
print('Optimal C value regression:', C_best)
print('Optimal gamma value regression:', gamma_best)

#Optimized model
SVMr_opt = svm.SVR(kernel = 'rbf', gamma = gamma_best, C = C_best)
cv_SVMr_tuned = cross_validate(SVMr_opt, Xtrain, ytrain.ravel(), cv = cv, n_jobs = -1, scoring = reg_scoring, return_train_score = False)
print('SVM regressor tuned MSE (CV): %0.3f (+/- %0.3f)' % (-cv_SVMr_tuned['test_MSE'].mean(), cv_SVMr_tuned['test_MSE'].std()*2))
print('SVM regressor tuned R2 (CV): %0.3f (+/- %0.3f)' % (cv_SVMr_tuned['test_R2'].mean(), cv_SVMr_tuned['test_R2'].std()*2))
print('SVM regressor tuned MAD (CV): %0.3f (+/- %0.3f)' % (-cv_SVMr_tuned['test_MAD'].mean(), cv_SVMr_tuned['test_MAD'].std()*2))
print('SVM regressor tuned accuracy (CV): %0.3f (+/- %0.3f)' % (-cv_SVMr_tuned['test_accuracy'].mean(), cv_SVMr_tuned['test_accuracy'].std()*2))
print('---')

#refit SVM regressor on training set and predict on unseen test data
print('-------------Best SVM regressor on test data------------')
pred_svm_reg = SVMr_opt.fit(Xtrain, ytrain).predict(Xtest)
print('MSE SVM test:', mean_squared_error(ytest, pred_svm_reg))
print('Max prediction SVM:', pred_svm_reg.max())
print('Min prediction SVM:', pred_svm_reg.min())
	
#confusion matrix
heatmap(confusion_matrix(ytest,np.rint(pred_svm_reg)),'SVM regressor confusion matrix, ' + wine_type, 'predicted', 'actual', np.unique(ytest), np.unique(ytest), True, format = '.0f', cmap = 'viridis', savefig = savefigs, figname = 'Images/SVM_reg_confusion_' + wine_type + '.png')

############ Decision tree classifier ############
print('-------------Decision tree (shallow) classifier------------')

#Shallow decision tree for visualization
dt_classifier = DecisionTreeClassifier(max_depth=3)
dt_classifier.fit(sc.inverse_transform(Xtrain),ytrain.ravel())
print('Shallow decision tree classifier accuracy train: %g' % accuracy_score(ytrain, dt_classifier.predict(sc.inverse_transform(Xtrain))))
print('Shallow decision tree classifier accuracy test: %g' % accuracy_score(ytest, dt_classifier.predict(sc.inverse_transform(Xtest))))

# Export the image to a dot file
ytrain_labels = [str(m) for m in np.unique(ytrain)]
export_graphviz(dt_classifier, out_file = 'Images/DT_classifier_simple_' + wine_type + '.dot', feature_names = feature_list, rounded = True, rotate = True, precision = 1, filled = True, class_names = ytrain_labels)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('Images/DT_classifier_simple_' + wine_type + '.dot')
# Write graph to a png file
graph.write_png('Images/DT_classifier_simple_' + wine_type + '.png')

#Comparative confusion matrices
pdb.set_trace()
#XGBoost regressor vs XGBoost classifier
heatmap(confusion_matrix(ytest,np.rint(pred_XGBoost_regressor_test))-confusion_matrix(ytest,pred_XGBoost_classifier_test),'XGBoost regressor-classifier confusion matrix, ' + wine_type, 'predicted', 'actual', np.unique(ytest), np.unique(ytest), True, format = '.0f', cmap = 'viridis', savefig = savefigs, figname = 'Images/XGBoost_reg_clas_confusion_' + wine_type + '.png')

#XGBoost regressor vs neural network regressor
heatmap(confusion_matrix(ytest,np.rint(pred_XGBoost_regressor_test))-confusion_matrix(ytest,np.rint(pred_nnKerasRegBest_test)),'XGBoost regressor - nn classifier confusion matrix, ' + wine_type, 'predicted', 'actual', np.unique(ytest), np.unique(ytest), True, format = '.0f', cmap = 'viridis', savefig = savefigs, figname = 'Images/XGBoost_reg_nn_clas_confusion_' + wine_type + '.png')

#Random forest regressor vs Ridge
heatmap(confusion_matrix(ytest,np.rint(pred_rf_regressor_test))-confusion_matrix(ytest,np.rint(pred_RidgeBest_test)),'Random forest regressor - Ridge regression confusion matrix, ' + wine_type, 'predicted', 'actual', np.unique(ytest), np.unique(ytest), True, format = '.0f', cmap = 'viridis', savefig = savefigs, figname = 'Images/RF_reg_nn_Ridge_confusion_' + wine_type + '.png')

#NN regressor vs NN classifier
heatmap(confusion_matrix(ytest,np.rint(pred_nnKerasRegBest_test))-confusion_matrix(ytest,pred_nnKerasBest_test),'NN regressor - NN classifier confusion matrix, ' + wine_type, 'predicted', 'actual', np.unique(ytest), np.unique(ytest), True, format = '.0f', cmap = 'viridis', savefig = savefigs, figname = 'Images/nn_reg_nn_clas_confusion_' + wine_type + '.png')
