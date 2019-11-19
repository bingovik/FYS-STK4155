import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, ShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, make_scorer, accuracy_score

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

originalclass = []
predictedclass = []
def classification_report_with_accuracy_score(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    return accuracy_score(y_true, y_pred) #return accuracy score

#######
#BINARY CLASSIFICATION
#######

y_b = y.copy()
y_b.values[y_b.values <= 9] = 0
y_b.values[y_b.values > 9] = 1

y1 = y_b.iloc[:,0]
y2 = y_b.iloc[:,1]
y3 = y_b.iloc[:,2]

Xtrain, Xtest, y1train, y1test, y2train, y2test, y3train, y3test = train_test_split(X_onehot, y1, y2, y3, test_size = 0.2)
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)

logreg = LogisticRegression(solver = 'lbfgs').fit(Xtrain, y3train)
#pred = logreg.predict(Xtest)
#print(classification_report(ytest_g3, pred))

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
scores_logreg = cross_val_score(logreg, Xtrain, y3train, cv = cv, scoring = make_scorer(classification_report_with_accuracy_score))

print('Class_report LOGROG', classification_report(originalclass, predictedclass))
print('CV accuracy score LOGREG: %0.3f (+/- %0.3f)' % (scores_logreg.mean(), scores_logreg.std()*2))


from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

def build_network(
            layer_sizes = (256,128,64,32,16),
            n_outputs = 1,
            batch_size = 64,
            epochs = 1000,
            loss = 'binary_crossentropy',
            alpha = 0,
            activation_function = 'relu',
            output_activation = 'sigmoid'
            ):
        model = Sequential()
        if isinstance(layer_sizes, int):
            layer_sizes = [layer_sizes]
        for layer_size in layer_sizes:
            model.add(Dense(layer_size, activation=activation_function,kernel_regularizer=regularizers.l2(alpha)))
        model.add(Dense(n_outputs, activation=output_activation))
        model.compile(loss=loss, optimizer='Adam', metrics=['accuracy'])
        return model

NN = KerasClassifier(build_fn=build_network)
scores_NN = cross_val_score(NN, Xtrain, y3train, cv = cv, scoring = make_scorer(classification_report_with_accuracy_score))

print('CV accuracy score NN: %0.3f (+/- %0.3f)' % (scores_NN.mean(), scores_NN.std()*2))

#GridSearchCV on Tensorflow/Keras neural network
parameters = {'layer_sizes':([256,128,64,32,16],[512,256,128,64,32,16],[1024,512,256,128]), 'activation_function':['sigmoid', 'relu'], 'alpha':[0, 0.01, 0.05, 0.1], 'epochs':[500,1000,1500]}
nnClassifier = KerasClassifier(build_fn=build_network, n_outputs=1, output_activation = 'sigmoid', loss = "binary_crossentropy", verbose=0)
clf = GridSearchCV(nnClassifier, parameters, scoring = 'accuracy', cv = 5, verbose = 8, n_jobs=-1)
clf.fit(Xtrain, y3train)
df_grid_nn_Keras = pd.DataFrame.from_dict(clf.cv_results_)

#order data into matrix
df_grid_nn_Keras, row_names_nn_Keras, col_names_nn_Keras = order_gridSearchCV_data(df_grid_nn_Keras, column_param = 'alpha')
#fit best Tensorflow/Keras model
print(clf.best_params_)
nnKerasBest = KerasClassifier(build_fn=build_network, n_outputs=1, output_activation = 'sigmoid', loss="binary_crossentropy",verbose=0)
nnKerasBest.set_params(**clf.best_params_)
nnKerasBest.fit(Xtrain, y3train)

#print metrics
print('Classification report for TensorFlow neural network:')
print(classification_report(y3test, nnKerasBest.model.predict_classes(Xtest)))
print('Accuracy: %g' % accuracy_score(y3test,nnKerasBest.model.predict_classes(Xtest)))
area_ratio_nn_Tensor = area_ratio(y3test, nnKerasBest.predict_proba(Xtest)[:,1], plot = True, title = 'Lift Keras/Tensorflow neural network', savefig=True, figname = 'Lift_Keras'+str(clf.best_params_)+fignamePostFix+'.png')
print('Area ratio: %g' % area_ratio_nn_Tensor)
print('Confusion matrix for TensorFlow neural network:')
conf = confusion_matrix(y3test, nnKerasBest.model.predict_classes(Xtest))
print(conf)
