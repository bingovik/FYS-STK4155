from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import pdb

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def standardize(X):
    means = np.mean(X, axis=0)
    stdevs = np.std(X, axis=0)
    #attempt to leave bias column unchanged if included
    ind_zero_stdev = stdevs == 0
    stdevs[ind_zero_stdev] = 1
    means[ind_zero_stdev] = 0
    X_standardized = (X - means)/stdevs
    return X_standardized, means, stdevs

def OLS_analytical(X, y):
    try:
        beta = np.linalg.inv(X.T@X)@X.T@y
    except np.linalg.LinAlgError as err:
        beta = np.linalg.pinv(X)@y
    return beta

def Ridge_analytical(X, y, _lambda):
    I = np.eye(X.shape[1])
    try:
        beta = np.linalg.inv(X.T@X + _lambda*I)@X.T@y
    except np.linalg.LinAlgError as err:
        beta = np.linalg.pinv(X.T@X + _lambda*I)@X.T@y
    return beta

def OLS_scikitlearn(X, y):
    clf = linear_model.LinearRegression(fit_intercept=False)
    clf.fit(X, y)
    return clf.coef_

def Ridge_scikitlearn(X, y, _lambda):
    clf = linear_model.Ridge(alpha=_lambda, fit_intercept=False)
    clf.fit(X, y)
    return clf.coef_

def Lasso(X, y, _lambda):
    clf = linear_model.Lasso(alpha=_lambda, fit_intercept=False)
    clf.fit(X, y)
    return clf.coef_
    
def get_stats(X, y, y_predict, sigma):
    try:
        var_beta = sigma**2*np.diag(np.linalg.inv(X.T@X))
    except np.linalg.LinAlgError as err:
        var_beta = sigma**2*np.diag(np.linalg.pinv(X.T@X))
    MSE = mean_squared_error(y, y_predict)
    R2 = r2_score(y, y_predict)
    return var_beta, MSE, R2

def surfPlot(x, y, z, xlabel = 'x', ylabel = 'y', zlabel = 'z' ):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                         linewidth=0, antialiased=False)
    # Customize the z axis.
    #ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()

def heatmap(data, title, xlabel, ylabel, xticks, yticks, annotation):
    ax = sns.heatmap(data, annot = annotation, linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(xticks, rotation=0)
    ax.set_yticklabels(yticks, rotation=0)
    plt.show()

def plot_several(x_data, y_data, colors, labels, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.set_title(title)    
    for i in range(x_data.shape[1]):
        plt.plot(x_data[:,i], y_data[:,i], label = labels[i])
    leg = ax.legend()
    plt.show()

def cv(reg_func, X, y, k, _lambda, X_test, y_test):
    X_test_orig = X_test
    k_size = round(len(X_train)/k)
    MSE_val = np.zeros(k)
    R2_val = np.zeros(k)
    y_predict_cv_val = np.zeros((k_size,k))
    y_predict_cv_test = np.zeros((len(X_test),k))
    y_cv_val = np.zeros((k_size,k))
    for i in range(k):
        test_ind = np.zeros(len(X), dtype = bool)
        test_ind[i*k_size:(i+1)*k_size] = 1
        X_cv_train = X[~test_ind]
        X_cv_val = X[test_ind]
        y_cv_train = y[~test_ind]
        y_cv_val[:,i] = y[test_ind]
        X_cv_train, mu, stdev = standardize(X_cv_train)
        X_cv_val = (X_cv_val - mu)/stdev
        X_test = (X_test_orig - mu)/stdev
        beta = reg_func(X_cv_train, y_cv_train, _lambda)
        y_predict_cv_val[:,i] = X_cv_val@beta
        y_predict_cv_test[:,i] = X_test@beta
        R2_val = r2_score(y_cv_val[:,i], y_predict_cv_val[:,i])
        MSE_val = mean_squared_error(y_cv_val[:,i], y_predict_cv_val[:,i])
    MSE_val = np.mean(MSE_val)
    R2_val = np.mean(R2_val)
    MSE_test = np.mean(np.mean((y_test[:,None] - y_predict_cv_test)**2, axis=0, keepdims=True))
    bias_test = np.mean((y_test[:,None] - np.mean(y_predict_cv_test, axis=1, keepdims=True))**2)
    variance_test = np.mean(np.var(y_predict_cv_test, axis=1, keepdims=True))
    return MSE_val, MSE_test, R2_val, bias_test, variance_test


#def get_data(input):
#    if input == 'Franke':

# Load the terrain
zz = imread('SRTM_data_Norway_2.tif')
# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(zz, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

x = np.arange(zz.shape[1])
y = np.arange(zz.shape[0])
yy, xx = np.meshgrid(x,y)
sigma = 0

#x = np.arange(0,1,0.05)
#y = np.arange(0,1,0.05)
#xx, yy = np.meshgrid(x,y)
#sigma = 0.02 #random noise std dev
#zz = FrankeFunction(xx, yy)
#seed(0)
#noise = np.resize(np.random.normal(0, sigma, xx.size),(len(x),len(y)))
#zz = zz + noise

surfPlot(xx, yy, zz)

poly_degree_max = 3
lambda_test_number = 8
lambda_tests = np.logspace(-7, 0, num=lambda_test_number)

z = np.ravel(zz)
MSE = np.zeros(poly_degree_max)
R2 = np.zeros(poly_degree_max)
var_beta_list = []
bias_test = np.zeros(poly_degree_max)
MSE_test = np.zeros(poly_degree_max)
MSE_cv_val_OLS = np.zeros(poly_degree_max)
MSE_cv_test_OLS = np.zeros(poly_degree_max)
R2_cv_val_OLS = np.zeros(poly_degree_max)
bias_cv_test_OLS = np.zeros(poly_degree_max)
variance_cv_test_OLS = np.zeros(poly_degree_max)
MSE_cv_val_Ridge = np.zeros((poly_degree_max, lambda_test_number))
MSE_cv_test_Ridge = np.zeros((poly_degree_max, lambda_test_number))
R2_cv_val_Ridge = np.zeros((poly_degree_max, lambda_test_number))
bias_cv_test_Ridge = np.zeros((poly_degree_max, lambda_test_number))
variance_cv_test_Ridge = np.zeros((poly_degree_max, lambda_test_number))
MSE_cv_val_Lasso = np.zeros((poly_degree_max, lambda_test_number))
MSE_cv_test_Lasso = np.zeros((poly_degree_max, lambda_test_number))
R2_cv_val_Lasso = np.zeros((poly_degree_max, lambda_test_number))
bias_cv_test_Lasso = np.zeros((poly_degree_max, lambda_test_number))
variance_cv_test_Lasso = np.zeros((poly_degree_max, lambda_test_number))
MSE_test_Ridge = np.zeros((poly_degree_max, lambda_test_number))
MSE_test_Lasso = np.zeros((poly_degree_max, lambda_test_number))
list_of_features = []
X_orig = np.vstack((np.ravel(xx), np.ravel(yy))).T

# running regressions using different polynomial fits up to poly_degree_max
for poly_degree in range(1,poly_degree_max+1):
    pdb.set_trace()
    
    # creating polynomials of degree poly_degree
    poly = PolynomialFeatures(poly_degree) #inlude bias = false
    X = poly.fit_transform(X_orig)
    features = poly.get_feature_names(['x','y'])
    list_of_features.append(features)

    # OLS on whole sample
    beta = OLS_scikitlearn(X, z)
    z_predict = X@beta
    var_beta, MSE[poly_degree - 1], R2[poly_degree - 1] = get_stats(X, z, z_predict, sigma)
    var_beta_list.append(list(var_beta))

    #surfPlot(xx, yy, z_predict.reshape((len(x),len(y))))

    # Splitting data into train and test sets
    X_train, X_test, z_train, z_test = train_test_split(X,z, test_size = 0.2)

    # feature scaling
    X_train, mu, stdev = standardize(X_train)
    X_test = (X_test - mu)/stdev

    # OLS evaluated on test set
    beta = OLS_scikitlearn(X_train, z_train)
    z_predict_test = X_test@beta
    var_beta, MSE_test[poly_degree-1], R2_test = get_stats(X_train, z_test, z_predict_test, sigma)

    # OLS estimates using cv
    MSE_cv_val_OLS[poly_degree-1], MSE_cv_test_OLS[poly_degree-1], R2_cv_val_OLS[poly_degree-1], bias_cv_test_OLS[poly_degree-1], variance_cv_test_OLS[poly_degree-1] = cv(Ridge_scikitlearn, X_train, z_train, 10, 0, X_test, z_test)

    for (i, lambda_test) in enumerate(lambda_tests):
        # Ridge and Lasso regression using cv and with different regularisation lambda
        MSE_cv_val_Ridge[poly_degree-1, i], MSE_cv_test_Ridge[poly_degree-1, i], R2_cv_val_Ridge[poly_degree-1, i], bias_cv_test_Ridge[poly_degree-1, i], variance_cv_test_Ridge[poly_degree-1, i] = cv(Ridge_scikitlearn, X_train, z_train, 10, lambda_test, X_test, z_test)
        MSE_cv_val_Lasso[poly_degree-1, i], MSE_cv_test_Lasso[poly_degree-1, i], R2_cv_val_Lasso[poly_degree-1, i], bias_cv_test_Lasso[poly_degree-1, i], variance_cv_test_Lasso[poly_degree-1, i] = cv(Lasso, X_train, z_train, 10, lambda_test, X_test, z_test)
        #calculating MSE using whole training sample (not cross-validating)
        beta = Ridge_scikitlearn(X_train, z_train, lambda_test)
        z_predict_test_Ridge = X_test@beta
        MSE_test_Ridge[poly_degree-1, i] = mean_squared_error(z_test, z_predict_test_Ridge)
        beta = Lasso(X_train, z_train, lambda_test)
        z_predict_test_Lasso = X_test@beta
        MSE_test_Lasso[poly_degree-1, i] = mean_squared_error(z_test, z_predict_test_Lasso)

# displaying heatmaps as functions of polynomial degree and regularization parameters
heatmap(MSE_cv_val_Ridge, 'Ridge validaton MSE', 'lambda', 'polynomial degree', np.around(lambda_tests, decimals=7), range(1,poly_degree_max+1), True)
heatmap(MSE_cv_test_Ridge, 'Ridge test MSE', 'lambda', 'polynomial degree', np.around(lambda_tests, decimals=7), range(1,poly_degree_max+1), True)
heatmap(MSE_cv_val_Lasso, 'Lasso validaton MSE', 'lambda', 'polynomial degree', np.around(lambda_tests, decimals=7), range(1,poly_degree_max+1), True)
heatmap(MSE_cv_test_Lasso, 'Lasso test MSE', 'lambda', 'polynomial degree', np.around(lambda_tests, decimals=7), range(1,poly_degree_max+1), True)
    
#plot bias and variance as function of polynomial degree
poly_degrees = np.repeat(np.arange(poly_degree_max)[:,None]+1, 3, axis=1)
plot_several(poly_degrees, 
    np.hstack((MSE_cv_test_OLS[:,None],bias_cv_test_OLS[:,None],variance_cv_test_OLS[:,None])), 
    ['r-', 'b-', 'g-'], ['MSE', 'bias^2', 'variance'], 'Polynomial degree', 'Franke', 'Bias-variance trade-off (OLS)')
plot_several(poly_degrees, np.hstack((MSE_cv_test_Ridge[:,4][:,None],bias_cv_test_Ridge[:,4][:,None],variance_cv_test_Ridge[:,4][:,None])), ['r-', 'b-', 'g-'], ['MSE', 'bias^2', 'variance'], 'Polynomial degree', 'Franke', 'Bias-variance trade-off (Ridge, lambda=something)')
plot_several(poly_degrees, np.hstack((MSE_cv_test_Lasso[:,4][:,None],bias_cv_test_Lasso[:,4][:,None],variance_cv_test_Lasso[:,4][:,None])), ['r-', 'b-', 'g-'], ['MSE', 'bias^2', 'variance'], 'Polynomial degree', 'Franke', 'Bias-variance trade-off (Lasso, lambda=something)')

# displaying OLS whole sample results
max_features = len(var_beta_list[-1])
for i in range(poly_degree_max):
    var_beta_list[i].extend([np.nan]*(max_features-len(var_beta_list[i])))
#var_beta_list = list(map(list, zip(*var_beta_list)))
var_beta_array = np.asarray(var_beta_list).T
conf_interval_beta = 1.96*np.sqrt(var_beta_array)
fig, axs = plt.subplots()
collabel = [('Degree ' + str(x+1)) for x in range(min(5,poly_degree_max))]
rowlabel = ['MSE','R2']
axs.set_title('OLS results whole dataset (no test sample)')
table_list_of_features = list_of_features[min(5-1,poly_degree_max)]
conf_interval_beta_text = ['Confidence: ' + i for i in table_list_of_features]
rowlabel.extend(conf_interval_beta_text)
axs.axis('off')
the_table = axs.table(cellText=np.around(np.vstack((MSE[0:min(5,poly_degree_max)], R2[0:min(5,poly_degree_max)], conf_interval_beta[0:len(list_of_features[min(5-1,poly_degree_max)]),0:min(5,poly_degree_max)])),decimals = 3),colLabels=collabel,rowLabels = rowlabel, loc='center')
plt.show()

# Derive bias variance formula
# Feature normalise data (already have function) - DONE
# Chart: show how Lasso sets certain betas to zero (pick degree 5)
# Chart: show
# Chart: show bias variance trade-off for OLS with noise=0.2 and data separation 0.1
# or with noise=0.02 and data separation 0.1 and polynomials up to degree 11
# Chart: plot best model using surfplot
# Chart: plot best model difference using surfplot
# Q: should not noise be a part of the total MSE sum?
# Q: For Ridge and Lasso, is var(beta) given by sigma²(X.T@X)^(-1) ?
# Q: Lasso convergence problem!
# Test results different than validation results (for cv)