
# coding: utf-8


from keras import applications

# build the VGG16 network
model = applications.VGG16(include_top=False,
                           weights='imagenet')


import os
import numpy as np
labels_list = open('list.txt').readlines()
name_list = [i.split()[0] for i in labels_list[6:]]
y = [i.split()[1] for i in labels_list[6:]]

print('image names', len(name_list), len(set(name_list)))
print('txt classes', len(y), len(set(y)))

# load image

from keras.preprocessing import image
images = [image.load_img(os.path.join('/rigel/edu/coms4995/datasets/pets/'+ name_list[i] + '.jpg'), target_size=(224, 224))
         for i in range(0,7349)]

# transform to numerical 
X = np.array([image.img_to_array(img) for img in images])

# get features
from keras.applications.vgg16 import preprocess_input
X_pre = preprocess_input(X)
features = model.predict(X_pre)

features.shape


features_ = features.reshape(7349, -1)
np.savetxt('feature', features_)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features_, y, stratify=y)


# train in Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(tol = 0.00001).fit(X_train, y_train)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, lr.predict(X_test))

# train in SGDCClassifier
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="hinge", penalty="l2", alpha=0.1)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
