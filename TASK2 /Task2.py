
# coding: utf-8


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold 
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline 
from keras.datasets import mnist
import keras

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# import data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

num_classes = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV

#grid search model with Dropout

def make_model1(optimizer="adam", hidden_size1=32, hidden_size1=32):
    model = Sequential([
        Dense(hidden_size1, input_shape=(784,), activation='relu'),
        Dense(hidden_size2, input_shape=(784,), activation='relu'),
        Dense(10, activation='softmax'),
    ])
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
    return model

clf = KerasClassifier(make_model1)

param_grid = {'epochs': [1, 5, 10, 20],  # epochs is fit parameter, not in make_model!
              'hidden_size1': [16,32,64,128],
              'hidden_size2': [16,32,64]}

grid = GridSearchCV(clf, param_grid=param_grid, cv=5)

grid.fit(X_train, y_train)
print("grid best score: {}".format(grid.best_score_))
print("grid best parameters: {}".format(grid.best_params_))
score = grid.score(X_test, y_test)
print(score)


# Best model without Dropout

model = Sequential([
    Dense(128, input_shape=(784,), activation='relu'),
    Dense(64, input_shape=(784,), activation='relu'),
    Dense(10, activation='softmax'),
])
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])


# In[52]:

history1 = model.fit(X_train, y_train, batch_size=128,
                    epochs=20, verbose=1, validation_split=.1)
score = model.evaluate(X_test, y_test, verbose=0)


# function to Plot history

def plot_history(logger):
    df = pd.DataFrame(logger.history)
    df[['acc', 'val_acc']].plot()
    plt.ylabel("accuracy")
    df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
    plt.ylabel("loss")


# lwarning curve without Dropout

df = pd.DataFrame(history1.history)
df[['acc', 'val_acc']].plot()
plt.ylabel("accuracy")
df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
plt.ylabel("loss")
plt.title('Without Drop')
plt.savefig('p1')
plt.show()


# grid search model with Dropout

from keras.layers import Dropout

def make_model2(optimizer="adam", hidden_size1=32,hidden_size2=32, drop=.5):
    model = Sequential([
        Dense(hidden_size1, input_shape=(784,), activation='relu'),
        Dropout(drop),
        Dense(hidden_size2, activation='relu'),
        Dropout(drop),
        Dense(10, activation='softmax'),
    ])
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
    return model

clf = KerasClassifier(make_model2)

param_grid = {'epochs': [1, 5, 10],  # epochs is fit parameter, not in make_model!
              'hidden_size': [32, 64, 256],
              'drop': [.3, .5, .7]}

grid = GridSearchCV(clf, param_grid=param_grid, cv=5)

grid.fit(X_train, y_train)
print("grid best score: {}".format(grid.best_score_))
print("grid best parameters: {}".format(grid.best_params_))
score = grid.score(X_test, y_test)
print(score)

# Best model with Dropout
from keras.layers import Dropout
model = Sequential([
    Dense(256, input_shape=(784,), activation='relu'),
    Dropout(.5),
    Dense(128, input_shape=(784,), activation='relu'),
    Dropout(.5),
    Dense(10, activation='softmax'),
])
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
history2 = model.fit(X_train, y_train, batch_size=128,epochs=20, verbose=1, validation_split=.1)



score = model.evaluate(X_test, y_test, verbose=0)
score

# learning curve with Dropout

df = pd.DataFrame(history2.history)
df[['acc', 'val_acc']].plot()
plt.ylabel("accuracy")
df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
plt.ylabel("loss")
plt.title('Without Drop')
plt.savefig('p2')
plt.show()




