import pdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve


def sigmoid(x):
	s = 1/(1+np.exp(-x))
	return s

def lift_chart(y_test,y_test_pred, plot = False, title = 'Lift chart', savefig = False, figname = ''):
	#cumulative gains/lift chart/area ratio
	sorting = np.argsort(-y_test_pred,axis = 0)
	y_test_pred_sorted = y_test_pred[np.squeeze(sorting)]
	y_test_sorted = y_test[np.squeeze(sorting)]
	curve_model = np.cumsum(y_test_sorted)
	curve_perfect_model = np.cumsum(-np.sort(-y_test, axis = 0))
	curve_no_model = np.linspace(curve_perfect_model[-1]/len(y_test),curve_perfect_model[-1],num=len(y_test))
	#fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
	#roc_auc = auc(fpr,tpr)
	area_model = auc(np.arange(len(y_test)), curve_model)
	area_perfect_model = auc(np.arange(len(y_test)), curve_perfect_model)
	area_no_model = auc(np.arange(len(y_test)), curve_no_model)
	area_ratio = (area_model - area_no_model)/(area_perfect_model - area_no_model)
	if plot:
		plot_several(np.repeat(np.arange(len(y_test))[:,None], 3, axis=1),
						np.hstack((curve_model[:,None],curve_perfect_model[:,None],curve_no_model[:,None])),
						['r-', 'b-'], ['Model', 'Perfect model', 'Baseline'],
						'Number of predictions', 'Cumulative number of defaults', title,
						savefig = savefig, figname = figname)
	return area_ratio

def log_loss(y_pred, y, beta):
	#accepts no categorical or sparse y
	n = len(y)
	log_loss = -(y.T@np.log(y_pred)+(1-y).T@np.log(1-y_pred))/n
	regularization_term = 0.5*_lambda*beta[1:].T@beta[1:]/n
	categorical_cross_entropy = log_loss[0,0] + regularization_term[0,0]
	return categorical_cross_entropy

def categorical_cross_entropy(y_pred, y):
	#accepts no categorical or sparse y, y_pred
	n = len(y)
	log_loss = -(y.T@np.log(y_pred)+(1-y).T@np.log(1-y_pred))/n
	return log_loss[0][0]

def cv(model, k, metric, X, y, X_test, y_test):
	''' 
	Performs k-fold cross validation on input design matrix x and target vector y.
	Predictions are also calculated for the separate test set (X_test, y_test) in
	order to estimate bias and variance
	'''
	k_size = int(np.floor(len(X)/k))
	metric_val = np.zeros(k)
	metric_test = np.zeros(k)
	y_predict_cv_val = np.zeros((k_size,k))
	y_predict_cv_test = np.zeros((len(X_test),k))
	y_cv_val = np.zeros((k_size,k))

	for i in range(k):
		#k-fold cross vaildation
		#splitting X into training and validation sets
		#no shuffling is done, assuming input is already shuffled.
		test_ind = np.zeros(len(X), dtype = bool)
		test_ind[i*k_size:(i+1)*k_size] = 1
		X_cv_train = X[~test_ind]
		X_cv_val = X[test_ind]
		y_cv_train = y[~test_ind]
		y_cv_val[:,i] = np.squeeze(y[test_ind])

		#train model and predict on validation and test sets
		model.train(X_cv_train, y_cv_train)
		y_predict_cv_val[:,i] = np.squeeze(model.predict(X_cv_val))
		y_predict_cv_test[:,i] = np.squeeze(model.predict(X_test))

		#calculating metric on the validation and test sets
		metric_val[i] = metric(y_predict_cv_val, y_cv_val)        
		metric_test[i] = metric(y_predict_cv_test, y_test)        

	metric_val = np.mean(metric_val)
	metric_test = np.mean(metric_test)

	#calculating MSE, variance and (bias + random noise) on the separate test set  
	MSE_test = np.mean(np.mean((y_test[:,None] - y_predict_cv_test)**2, axis=0, keepdims=True))    
	variance_test = np.mean(np.var(y_predict_cv_test, axis=1))
	bias_test_plus_noise = np.mean((y_test[:,None] - np.mean(y_predict_cv_test, axis=1, keepdims=True))**2)

	return metric_val, metric_test, MSE_test, variance_test, bias_test_plus_noise

def plot_several(x_data, y_data, colors, labels, xlabel, ylabel, title, savefig = False, figname = ''):
    fig, ax = plt.subplots()
    plt.xlabel(xlabel, fontsize = 9)
    plt.ylabel(ylabel, fontsize = 9)
    ax.set_title(title, fontsize = 11)    
    for i in range(x_data.shape[1]):
        plt.plot(x_data[:,i], y_data[:,i], label = labels[i])
    leg = ax.legend()
    if savefig: plt.savefig(figname, dpi=300, bbox_inches='tight') 
    plt.show()