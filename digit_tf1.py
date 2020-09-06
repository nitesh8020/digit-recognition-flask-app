#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 19:05:14 2020

@author: nitesh
"""


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt

# the data, split between train and test sets



(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape,y_train.shape)

X_train = x_train.reshape(-1,28,28,1)
X_test = x_test.reshape(-1,28,28,1)
Y_train = keras.utils.to_categorical(y_train, 10)
Y_test = keras.utils.to_categorical(y_test, 10)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255
X_test = X_test/255

# plt.imshow(X_train[1].reshape(28,28),cmap='gray')
# print(Y_train[1])

mini_batch = 256
n_epoch = 10

img_shape = (28,28,1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=img_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])



hist = model.fit(X_train, Y_train,batch_size=mini_batch,epochs=n_epoch,verbose=1,validation_data=(X_test, Y_test))
print("The model has successfully trained")
model.save('mnist.h5')
print("Saving the model as mnist.h5")

model = keras.models.load_model('mnist.h5')
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
