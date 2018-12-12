# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 13:21:41 2018

@author: Imen
"""
import keras
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from tensorflow.python.keras.layers import Convolution2D as Conv2D
from tensorflow.python.keras.layers import MaxPooling2D,Dense,Flatten
import cv2
from tensorflow.python.keras.optimizers import SGD
from sklearn.multiclass import OneVsRestClassifier
from PIL import Image
import glob
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from tensorflow.python.keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers.advanced_activations import LeakyReLU

mypath='C:/Users/Imen/Desktop/MIT/MLpython/yalefaces'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
#change extension of files
'''
files = os.listdir(path)
for file in files:
    os.rename(os.path.join(mypath, file), os.path.join(mypath,file+'.jpg'))
'''
#construct the label list
listmot=[]
for mot in onlyfiles:
    listmot.append(mot.split('.'))
y=[]
for mot in listmot:
    y.append(mot[0].split("t")[1])
lc= LabelEncoder()
y = lc.fit_transform(y)
image_list = []
image_toarray=[]
for filename in glob.glob('C:/Users/Imen/Desktop/MIT/MLpython/yalefaces/*.jpg'): 
    im=Image.open(filename)
    image_list.append(im)
    image_toarray.append(np.array(im))
image_toarray= np.array(image_toarray)
X_train,X_test,y_train,y_test=train_test_split(image_toarray,y,train_size=0.66)

#plt.imshow(X_train[0,:,:], cmap='gray')
#plt.imshow(X_test[0,:,:], cmap='gray')
X_train=X_train.reshape(-1,243,320,1)
X_test=X_test.reshape(-1,243,320,1)
X_train= X_train.astype('float32')
X_test= X_test.astype('float32')
X_train= X_train / 255.
X_test= X_test / 255.

#y_train=keras.utils.to_categorical(y_train)
#y_test=keras.utils.to_categorical(y_test)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(243,320,1),padding='same'))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(Dense(15, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=SGD(),
              metrics=['accuracy'])
model.fit(X_train,y_train,epochs=35,batch_size=50)

test_eval =model.evaluate(X_test, y_test)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
predicted_classes =model.predict(X_test)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
correct = np.where(predicted_classes==y_test)[0]
print( "from 57 Found %d correct labels" % len(correct))

