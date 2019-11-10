import pdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, mean_squared_error, r2_score
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

def build_network(layer_sizes=[50,20], n_outputs = 2,
                batch_size=32,
                epochs=10,
                optimizer="Adam",
                loss="categorical_crossentropy",
                alpha = 0.0,
                activation_function = 'relu',
                output_activation = 'softmax'
                ):
        model = Sequential()
        #model.add(BatchNormalization())
        if isinstance(layer_sizes, int):
            layer_sizes = [layer_sizes]
        for layer_size in layer_sizes:
            model.add(Dense(layer_size, activation=activation_function,kernel_regularizer=regularizers.l2(alpha)))
        model.add(Dense(n_outputs, activation=output_activation))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        return model

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def surfPlot(x, y, z, xlabel = 'x', ylabel = 'y', zlabel = 'z', savefig = False, figname = ''):
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
    #s = (x>=0)/(1+np.exp(-x)) + (x<0)*exp(x)/(1+np.exp(x)) 
    return s

def relu(x):
    return np.reshape(x*(x>0),x.shape)

def softmax(x):
    #made numerically stable by subtracting max values
    exp_term = np.exp(x - np.amax(x,axis = 1, keepdims = True))
    return exp_term / np.sum(exp_term, axis=1, keepdims=True)

def area_ratio(y_test,y_test_pred, plot = False, title = 'Lift chart', savefig = False, figname = ''):
    #cumulative gains/lift chart/area ratio
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
    #accepts no categorical or sparse y
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

def cv(model, k, metric, X, y, X_test, y_test, max_iter):
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
        model.fit(X_cv_train, y_cv_train, max_iter = max_iter)
        y_predict_cv_val = model.predict(X_cv_val)
        y_predict_cv_test[:,i] = np.squeeze(model.predict(X_test))

        #calculating metric on the validation and test sets
        metric_val[i] = metric(y_predict_cv_val, y_cv_val)
        metric_test[i] = metric(y_predict_cv_test[:,i][:,None], y_test)

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

def heatmap(data, title, xlabel, ylabel, xticks, yticks, annotation, savefig = False, figname = ''):
    ax = sns.heatmap(data, fmt = '.3f', annot = annotation, linewidth=0.5)
    sns.set(font_scale=0.64)
    ax.set_title(title, fontsize = 11)
    ax.set_xlabel(xlabel, fontsize = 9)
    ax.set_ylabel(ylabel, fontsize = 9)
    ax.set_xticklabels(xticks, rotation=90, fontsize = 9)
    ax.set_yticklabels(yticks, rotation=0, fontsize = 9)
    ax.set_ylim(len(data)+0.25, -0.25)
    if savefig: plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.show()

def order_gridSearchCV_data(pandas_df, column_param = '_lambda'):
    col_values = pandas_df['param_'+column_param]
    uniqueColumns = np.asarray(col_values.drop_duplicates())
    data_array = np.empty((int(len(pandas_df.index)/len(uniqueColumns)),len(uniqueColumns)))
    for i, cols in enumerate(uniqueColumns):
        df_temp = pandas_df.loc[pandas_df['param_'+column_param]==cols]
        row_names = df_temp['params']
        data_array[:,i] = np.asarray(df_temp['mean_test_score'])
    row_names = row_names.tolist()
    for w in row_names:
        del w[column_param]
    row_names = [str(w) for w in row_names]
    row_names = [w.replace('activation_function', 'a_func') for w in row_names]
    row_names = [w.replace('layer_sizes', 'h_layers') for w in row_names]
    row_names = [w.replace('n_hidden_neurons', 'nodes') for w in row_names]
    row_names = [w.replace('alpha', '\u03BB') for w in row_names] #alpha with lambda haha
    row_names = [w.replace('_lambda', '\u03BB') for w in row_names]
    row_names = [w.replace('lmbd', '\u03BB') for w in row_names]
    row_names = [w.replace(' ', '') for w in row_names]
    row_names = [w.replace('sigmoid', 'sigm') for w in row_names]
    row_names = [w.replace('{', '') for w in row_names]
    row_names = [w.replace('}', '') for w in row_names]
    return data_array, row_names, uniqueColumns

def order_grid_search_data(param_grid, param_grid_obj, val_acc, column_param = 'alpha'):
    col_names = param_grid[column_param]
    row_names = [None]*len(param_grid_obj)
    datadata = np.zeros((len(param_grid_obj),len(col_names)))
    for i, g in enumerate(param_grid_obj):
        for j, alph in enumerate(col_names):
            if param_grid_obj[i][column_param] == alph:
                datadata[i,j] = val_acc[i]
                dict_temp = param_grid_obj[i]
                del dict_temp[column_param]
                row_names[i] = str(dict_temp)
    dd = np.ma.masked_equal(datadata[:,0],0)
    data_array = np.empty((sum(~dd.mask),len(col_names))) 
    for c in range(datadata.shape[1]):
        datacolumn = np.ma.masked_equal(datadata[:,c],0)
        data_array[:,c] = datacolumn.compressed()
    ind_list = [i for i, x in enumerate(dd.mask) if not x]
    row_names = [row_names[i] for i in ind_list]
    row_names = [w.replace('activation_function', 'a_func') for w in row_names]
    row_names = [w.replace('layer_sizes', 'nodes') for w in row_names]
    row_names = [w.replace('alpha', '\u03BB') for w in row_names] #alpha with lambda...
    row_names = [w.replace(' ', '') for w in row_names]
    row_names = [w.replace('sigmoid', 'sigm') for w in row_names]
    row_names = [w.replace('{', '') for w in row_names]
    row_names = [w.replace('}', '') for w in row_names]
    return data_array, row_names, col_names

def cross_validation_OLS(x, y, k):
    n = len(x)

    indexes = np.arange(y.shape[0])
    np.random.shuffle(indexes)
    x = x[indexes]
    y = y[indexes]

    r2_train = []
    r2_test = []
    mse_train = []
    mse_test = []

    for i in range(k):
        x_train = np.concatenate((x[:int(i*n/k)], x[int((i + 1)*n/k): ]), axis = 0)
        x_test = x[int(i*n/k):int((i + 1)*n/k)]
        y_train = np.concatenate((y[:int(i*n/k)], y[int((i + 1)*n/k): ]), axis = 0)
        y_test = y[int(i*n/k):int((i + 1)*n/k)]

        beta = np.linalg.pinv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
        ytilde = x_train @ beta
        ypredict = x_test @ beta

        mse_train.append(mean_squared_error(y_train, ytilde))
        mse_test.append(mean_squared_error(y_test, ypredict))

        r2_train.append(r2_score(y_train, ytilde))
        r2_test.append(r2_score(y_test, ypredict))

    r2_train = np.array(r2_train)
    r2_test = np.array(r2_test)
    mse_train = np.array(mse_train)
    mse_test = np.array(mse_test)

    return mse_train, mse_test, r2_train, r2_test