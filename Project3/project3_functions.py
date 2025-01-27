from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import pdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, mean_squared_error, r2_score, accuracy_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import tensorflow as tf
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from keras import regularizers
import seaborn as sns
from sklearn.utils import resample

def accuracy_from_regression(y,y_pred):
    #rounds predictions no nearest integer and calculates
    #ratio of correct predictions (accuracy)
    return accuracy_score(y,np.rint(y_pred))

def MAD(y_true, y_pred):
    #calculates mean absolute deviation
    return np.mean(np.abs(y_true.ravel() - y_pred.ravel()))

def build_network(layer_sizes=[50,20],
                n_outputs = 2,
                batch_size=32,
                epochs=10,
                optimizer="Adam",
                loss="categorical_crossentropy",
                alpha = 0.0,
                activation_function = 'relu',
                output_activation = 'softmax'
                ):
        #function to construct a network using Keras/Tensorflow
        model = Sequential()
        #model.add(BatchNormalization())
        if isinstance(layer_sizes, int):
            layer_sizes = [layer_sizes]
        for layer_size in layer_sizes:
            model.add(Dense(layer_size, activation=activation_function,kernel_regularizer=regularizers.l2(alpha)))
        if output_activation != None:
            model.add(Dense(n_outputs, activation=output_activation))
        else:
            model.add(Dense(n_outputs))

        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        return model

def surfPlot(x, y, z, xlabel = 'x', ylabel = 'y', zlabel = 'z', savefig = False, figname = ''):
    #function collecting commands to show a surfplot using MatPlotLib
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                         linewidth=0, antialiased=False)
    # Customize the z axis.
    #ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(xlabel, fontsize = 9)
    ax.set_ylabel(ylabel, fontsize = 9)
    ax.set_zlabel(zlabel, fontsize = 9)
    if savefig: plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.show()

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def relu(x):
    return np.reshape(x*(x>0),x.shape)

def softmax(x):
    #made numerically stable by subtracting max values
    exp_term = np.exp(x - np.amax(x,axis = 1, keepdims = True))
    return exp_term / np.sum(exp_term, axis=1, keepdims=True)

def area_ratio(y_test,y_test_pred, plot = False, title = 'Lift chart', savefig = False, figname = ''):
    #calculate cumulative gains/lift chart/area ratio
    #boolean variable 'plot' controls whether to show the cumulative gains chart
    #plot can be save by setting 'savefig' to True and providing a filename to the 'figname' variable
    sorting = np.argsort(-y_test_pred,axis = 0)
    y_test_pred_sorted = y_test_pred[np.squeeze(sorting)]
    y_test_sorted = y_test[np.squeeze(sorting)]
    curve_model = np.cumsum(y_test_sorted)
    curve_perfect_model = np.cumsum(-np.sort(-y_test, axis = 0))
    curve_no_model = np.linspace(curve_perfect_model[-1]/len(y_test),curve_perfect_model[-1],num=len(y_test))
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
    #loss function. Accepts no categorical or sparse y
    n = len(y)
    categorical_cross_entropy = -(y.T@np.log(y_pred)+(1-y).T@np.log(1-y_pred))/n
    regularization_term = 0.5*_lambda*beta[1:].T@beta[1:]/n
    log_loss = categorical_cross_entropy[0,0] + regularization_term[0,0]
    return log_loss

def categorical_cross_entropy(y_pred, y):
    #accepts no categorical or sparse y, y_pred
    n = len(y)
    categorical_cross_entropy = -(y.T@np.log(y_pred)+(1-y).T@np.log(1-y_pred))/n
    return categorical_cross_entropy[0][0]

def crossVal(model, k, metric, X, y, X_test, y_test):
    '''
    Performs k-fold cross validation on input design matrix x and target vector y.
    Predictions are also calculated for the separate test set (X_test, y_test) in
    order to estimate bias and variance
    '''
    k_size = int(np.floor(len(X)/k))
    metric_val = np.zeros(k)
    metric_test = np.zeros(k)
    y_predict_cv_test = np.zeros((len(X_test),k))

    for i in range(k):
        #k-fold cross vaildation
        #splitting X into training and validation sets
        #no shuffling is done, assuming input is already shuffled.
        test_ind = np.zeros(len(X), dtype = bool)
        test_ind[i*k_size:(i+1)*k_size] = 1
        X_cv_train = X[~test_ind]
        X_cv_val = X[test_ind]
        y_cv_train = y[~test_ind]
        y_cv_val = y[test_ind]

        #train model and predict on validation and test sets
        model.fit(X_cv_train, y_cv_train)
        y_predict_cv_val = model.predict(X_cv_val)
        y_predict_cv_test[:,i] = np.squeeze(model.predict(X_test))

        #calculating metric on the validation and test sets
        metric_val[i] = metric(y_predict_cv_val, y_cv_val)
        metric_test[i] = metric(y_predict_cv_test[:,i][:,None], y_test)

    metric_val = np.mean(metric_val)
    metric_test = np.mean(metric_test)

    #calculating MSE, variance and (bias + random noise) on the separate test set
    MSE_test = np.mean(np.mean((y_test - y_predict_cv_test)**2, axis=0, keepdims=True))
    variance_test = np.mean(np.var(y_predict_cv_test, axis=1))
    bias_test_plus_noise = np.mean((y_test - np.mean(y_predict_cv_test, axis=1, keepdims=True))**2)

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

def heatmap(data, title, xlabel, ylabel, xticks, yticks, annotation, format = '.3f', cmap = None, savefig = False, figname = ''):
    ax = sns.heatmap(data, fmt = format, cmap=cmap, annot = annotation, linewidth=0.5)
    sns.set(font_scale=0.64)
    ax.set_title(title, fontsize = 11)
    ax.set_xlabel(xlabel, fontsize = 9)
    ax.set_ylabel(ylabel, fontsize = 9)
    ax.set_xticklabels(xticks, rotation=90, fontsize = 9)
    ax.set_yticklabels(yticks, rotation=0, fontsize = 9)
    ax.set_ylim(len(data)+0.25, -0.25)
    if savefig: plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.show()

def order_gridSearchCV_data(pandas_df, column_param = '_lambda', score = 'mean_test_score'):
    #function takes a pandas dataframe containing output from Scikit-Learn's gridsearchcv method
    #and transforms it to a 2-dimensional array. A variable of choice 'column_param' will be
    #distributed along the columns while the rest of the parameters will be along the rows.
    col_values = pandas_df['param_'+column_param]
    uniqueColumns = np.asarray(col_values.drop_duplicates())
    data_array = np.empty((int(len(pandas_df.index)/len(uniqueColumns)),len(uniqueColumns)))
    df_temp = pandas_df.loc[pandas_df['param_'+column_param]==min(uniqueColumns)]
    row_names = df_temp['params'].tolist()
    for i, cols in enumerate(uniqueColumns):
        df_temp = pandas_df.loc[pandas_df['param_'+column_param]==cols]
        data_array[:,i] = np.asarray(df_temp[score])
    for w in row_names:
        if column_param in w.keys(): del w[column_param]
    row_names = [str(w) for w in row_names]
    row_names = [w.replace('activation_function', 'a_func') for w in row_names]
    row_names = [w.replace('layer_sizes', 'h_layers') for w in row_names]
    row_names = [w.replace('n_hidden_neurons', 'nodes') for w in row_names]
    row_names = [w.replace('alpha', '\u03BB') for w in row_names] #alpha with lambda....
    row_names = [w.replace('_lambda', '\u03BB') for w in row_names]
    row_names = [w.replace('lmbd', '\u03BB') for w in row_names]
    row_names = [w.replace(' ', '') for w in row_names]
    row_names = [w.replace('sigmoid', 'sigm') for w in row_names]
    row_names = [w.replace('{', '') for w in row_names]
    row_names = [w.replace('}', '') for w in row_names]
    return data_array, row_names, uniqueColumns

def bootstrap_bias_variance_MSE(model, X, y, n_boostraps, X_test, y_test):
    #function uses bootstrapping to calculate bias/variance from 'model' applied to 'X'
    #and 'y' and tested on 'X__test' and 'y_test'
    y_pred = np.empty((y_test.shape[0], n_boostraps))
    for i in range(n_boostraps):
        x_, y_ = resample(X, y)
        # Fit model and then predict on the same test data each time.
        model.fit(x_,y_)
        y_pred[:, i] = model.predict(X_test).ravel()
    error = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
    bias = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(y_pred, axis=1, keepdims=True) )
    return error, bias, variance