#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 21:52:03 2020

@author: nitesh
"""
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import os


# the data, split between train and test sets
print(tf.__version__)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape,y_train.shape)

X_train = x_train.reshape(-1,28,28,1)
X_test = x_test.reshape(-1,28,28,1)
Y_train = tf.keras.utils.to_categorical(y_train, 10)
Y_test = tf.keras.utils.to_categorical(y_test, 10)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255
X_test = X_test/255

# X_train = X_train[..., tf.newaxis]
# X_test = X_test[..., tf.newaxis]
# plt.imshow(X_train[1].reshape(28,28),cmap='gray')
# print(Y_train[1])

mini_batch = 128
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

model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer='adam', metrics=['accuracy'])

print(X_train.shape, X_test.shape,Y_train.shape)
# model.fit(X_train, Y_train,batch_size = mini_batch,epochs=n_epoch,verbose=1)
# print("The model has successfully trained")
# model.save('mnist1.h5')
# print("Saving the model as mnist.h5")

model = tf.keras.models.load_model('mnist1.h5')
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])