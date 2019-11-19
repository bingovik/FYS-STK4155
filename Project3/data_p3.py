import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, ShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, make_scorer, accuracy_score

from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


dfr = pd.read_csv('./Data/winequality-red.csv', sep = ';')
dfw = pd.read_csv('./Data/winequality-red.csv', sep = ';')

# Create design matrix and target variable
X = dfr.iloc[:, :11].values
y = dfr.iloc[:, 11].values
y_c = dfr.iloc[:, 11]

#One hot encoding, it automatically does the cat/object variables
sc = StandardScaler()

originalclass = []
predictedclass = []
def classification_report_with_accuracy_score(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    return accuracy_score(y_true, y_pred) #return accuracy score

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)


#######
#BINARY CLASSIFICATION
#######
"""
y_b = y.copy()
y_b[y_b <= 5] = 0
y_b[y_b > 5] = 1

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_b, test_size = 0.2)
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)
X_ = sc.transform(X)

logreg = LogisticRegression(solver = 'lbfgs').fit(Xtrain, ytrain)
#pred = logreg.predict(Xtest)
#print(classification_report(ytest_g3, pred))

scores_logreg = cross_val_score(logreg, X_, y_b, cv = cv, scoring = make_scorer(classification_report_with_accuracy_score))

print('Class_report LOGROG', classification_report(originalclass, predictedclass))
print('CV accuracy score LOGREG: %0.3f (+/- %0.3f)' % (scores_logreg.mean(), scores_logreg.std()*2))

def build_network(
            layer_sizes = (128,64,32,16),
            n_outputs = 1,
            batch_size = 64,
            epochs = 1000,
            loss = 'binary_crossentropy', #or accuracy
            alpha = 0,
            activation_function = 'relu',
            output_activation = 'sigmoid'
            ):
        model = Sequential()
        if isinstance(layer_sizes, int):
            layer_sizes = [layer_sizes]
        for layer_size in layer_sizes:
            model.add(Dense(layer_size, activation=activation_function, kernel_regularizer=regularizers.l2(alpha)))
        model.add(Dense(n_outputs, activation=output_activation))
        model.compile(loss=loss, optimizer='Adam', metrics=['accuracy'])
        return model

NN = KerasClassifier(build_fn=build_network)
scores_NN = cross_val_score(NN, X_, y_b, cv = cv)

print(scores_NN)

print('CV accuracy score NN: %0.3f (+/- %0.3f)' % (scores_NN.mean(), scores_NN.std()*2))


model = Sequential()
#model.add(Dense(128, activation='relu'))
#model.add(Dense(64, activation='relu'))
#model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(Xtrain, ytrain, validation_split=0.3, epochs=100, batch_size=16)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("./Images/acc_hist.png")
plt.show()

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./Images/loss_hist.png')
plt.show()
"""

#######
#CATEGORICAL CLASSIFICATION
#######

onehotencoder = OneHotEncoder(categories = "auto")

y_c = pd.get_dummies(y_c).values

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_c, test_size = 0.2)
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)
X_ = sc.transform(X)

def build_network(
            layer_sizes = (128,64,32),
            n_outputs = 6,
            batch_size = 32,
            epochs = 20,
            loss = 'categorical_crossentropy', #or accuracy
            alpha = 0,
            activation_function = 'relu',
            output_activation = 'softmax'
            ):
        model = Sequential()
        if isinstance(layer_sizes, int):
            layer_sizes = [layer_sizes]
        for layer_size in layer_sizes:
            model.add(Dense(layer_size, activation=activation_function, kernel_regularizer=regularizers.l2(alpha)))
        model.add(Dense(n_outputs, activation=output_activation))
        model.compile(loss=loss, optimizer='Adam', metrics=['accuracy'])
        return model

# Fit the model
history = build_network().fit(Xtrain, ytrain, validation_split=0.3, epochs=100, batch_size=32)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("./Images/acc_hist_cat.png")
#plt.show()

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./Images/loss_hist_cat.png')
#plt.show()

#evaluate model
NN = KerasClassifier(build_fn=build_network)
scores_NN = cross_val_score(NN, X_, y_c, cv = cv)

print(scores_NN)

print('CV accuracy score NN: %0.3f (+/- %0.3f)' % (scores_NN.mean(), scores_NN.std()*2))
