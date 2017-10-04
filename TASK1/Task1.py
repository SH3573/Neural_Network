#!/bin/sh
# coding: utf-8

# In[2]:

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold 
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline 


# In[3]:

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# In[4]:

from sklearn import datasets
# load iris data
iris = datasets.load_iris()
X = iris.data
Y = iris.target
# convert integers to dummy variables (i.e. one hot encoded)
y = np_utils.to_categorical(Y)
# split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)


# In[ ]:

# Function to create model, required for KerasClassifier
# define baseline model

def make_model(hidden_size=3,act='relu'):
    # create model
    model = Sequential()
    # layer 1
    model.add(Dense(hidden_size, input_dim=4, kernel_initializer='normal', activation=act))
    # layer 2
    model.add(Dense(hidden_size, kernel_initializer='normal', activation=act))
    model.add(Dense(3, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
 
clf = KerasClassifier(build_fn=make_model)
param_grid = {'epochs': [50, 100, 200],  
              'batch_size': [5, 10, 15],
              'hidden_size': [4, 5, 6],
              'act': ['relu','tanh']} 

grid = GridSearchCV(clf, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print("grid best score: {}".format(grid.best_score_))
print("grid best parameters: {}".format(grid.best_params_))
score = grid.score(X_test, y_test)
print(score)






