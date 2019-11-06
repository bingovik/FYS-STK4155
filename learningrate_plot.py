# Which value for eta & lambda
eta_vals = [1e-6, 1e-4, 0.005, 1e-2, 0.05] #np.logspace(-2, 1, 7)
lmbd_vals = [0, 1e-5, 1e-4, 1e-3, 0.01, 0.1] #np.logspace(-2, 1, 7)

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
k = 5

test_score = np.zeros((len(eta_vals), k))
for i, eta in enumerate(eta_vals):
    DNN_ = NeuralNetworkRegressor(activation_function = activation_function, n_hidden_neurons = layer_size, epochs = epochs, batch_size = batch_size,
                                     eta=eta, lmbd=0)
    cv_DNN_ = cross_validate(DNN_, X, z[:,None], cv = cv, scoring = 'neg_mean_squared_error', return_train_score = False, n_jobs = -1)
    #train_score[i] = np.mean(cv_DNN_['train_score']) #only if return_train_score = True
    test_score[i] = np.mean(cv_DNN_['test_score'])

plt.figure()
plt.plot(eta_vals, test_score, 'r--', label = 'test_score')
plt.xlabel('eta_vals')
plt.ylabel('MSE')
plt.xticklabels(eta_vals)
plt.title('')
plt.legend()
plt.show()
