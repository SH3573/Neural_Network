import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import numpy as np
import scipy.io as sio

# Read data
train = sio.loadmat('train_32x32.mat')
test = sio.loadmat('test_32x32.mat')

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 32, 32

# Train/test
x_train = train['X']
y_train = train['y'].reshape(73257,)
x_test = test['X']
y_test = test['y'].reshape(26032,)

# Switch axes
X_train = np.expand_dims(x_train,axis=0)
X_train = X_train.swapaxes(4,0)
X_train = X_train.reshape(73257, 32, 32, 3)
X_train = X_train/255.0
X_test = np.expand_dims(x_test,axis=0)
X_test = X_test.swapaxes(4,0)
X_test = np.squeeze(X_test, axis=(4,))
X_test = X_test/255.0
input_shape = (img_rows, img_cols, 3)

y_train[y_train == 10] = 0
y_test[y_test == 10] = 0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Model 1
# Achieve 85% test-set accuracy with a base model
cnn = Sequential()
cnn.add(Conv2D(16, kernel_size=(6, 6),
               activation='relu',
               input_shape=input_shape))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(32, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(32, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())
cnn.add(Dense(512, activation='relu'))
cnn.add(Dense(num_classes, activation='softmax'))
cnn.compile("adam", "categorical_crossentropy", metrics=['accuracy'])

cnn.fit(X_train, y_train, batch_size=128, epochs=30, verbose=1, validation_split=0.1)
score = cnn.evaluate(X_test, y_test)
print("Test loss: {:.3f}".format(score[0]))
print("Test Accuracy: {:.3f}".format(score[1]))

# Epoch 26/30
# 65931/65931 [==============================] - 6s - loss: 0.2900 - acc: 0.9110 - val_loss: 0.4064 - val_acc: 0.8785
# Epoch 27/30
# 65931/65931 [==============================] - 6s - loss: 0.2836 - acc: 0.9115 - val_loss: 0.4111 - val_acc: 0.8792
# Epoch 28/30
# 65931/65931 [==============================] - 6s - loss: 0.2834 - acc: 0.9111 - val_loss: 0.4068 - val_acc: 0.8832
# Epoch 29/30
# 65931/65931 [==============================] - 6s - loss: 0.2744 - acc: 0.9149 - val_loss: 0.4084 - val_acc: 0.8795
# Epoch 30/30
# 65931/65931 [==============================] - 6s - loss: 0.2726 - acc: 0.9153 - val_loss: 0.4068 - val_acc: 0.8808
# 25568/26032 [============================>.] - ETA: 0s
# Test loss: 0.447
# Test Accuracy: 0.876

# Model 2: using Batch Normalization
# BONUS: accuracy >= 90%
cnn_small_bn = Sequential()
cnn_small_bn.add(Conv2D(258, kernel_size=(12, 12),
                        input_shape=input_shape))
cnn_small_bn.add(BatchNormalization(momentum=0.5))
cnn_small_bn.add(Activation("relu"))
cnn_small_bn.add(MaxPooling2D(pool_size=(2, 2)))
cnn_small_bn.add(Conv2D(128, (3, 3)))
cnn_small_bn.add(BatchNormalization())
cnn_small_bn.add(Activation("relu"))
cnn_small_bn.add(MaxPooling2D(pool_size=(2, 2)))
cnn_small_bn.add(Conv2D(32, (3, 3)))
cnn_small_bn.add(BatchNormalization())
cnn_small_bn.add(Activation("relu"))
cnn_small_bn.add(MaxPooling2D(pool_size=(2, 2)))
cnn_small_bn.add(Flatten())
cnn_small_bn.add(Dense(512, activation='tanh'))
cnn_small_bn.add(Dense(32, activation='relu'))
cnn_small_bn.add(Dense(num_classes, activation='softmax'))

cnn_small_bn.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_cnn_small_bn = cnn_small_bn.fit(X_train, y_train, batch_size=128, epochs=30, verbose=1, validation_split=.1)

score_n = cnn_small_bn.evaluate(X_test, y_test, verbose=0)
print("Test loss with Batch Normalization: {:.3f}".format(score_n[0]))
print("Test Accuracy with Batch normalization: {:.3f}".format(score_n[1]))

# Last 5 epochs:
# Epoch 26/30
# 65931/65931 [==============================] - 33s - loss: 0.1083 - acc: 0.9663 - val_loss: 0.3493 - val_acc: 0.9163
# Epoch 27/30
# 65931/65931 [==============================] - 33s - loss: 0.1084 - acc: 0.9668 - val_loss: 0.4163 - val_acc: 0.9016
# Epoch 28/30
# 65931/65931 [==============================] - 33s - loss: 0.1065 - acc: 0.9665 - val_loss: 0.3838 - val_acc: 0.9053
# Epoch 29/30
# 65931/65931 [==============================] - 33s - loss: 0.1040 - acc: 0.9675 - val_loss: 0.3742 - val_acc: 0.9109
# Epoch 30/30
# 65931/65931 [==============================] - 33s - loss: 0.0895 - acc: 0.9721 - val_loss: 0.3874 - val_acc: 0.9106
# Test loss with Batch Normalization: 0.436
# Test Accuracy with Batch normalization: 0.900
