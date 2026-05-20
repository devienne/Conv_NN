#!/usr/bin/env python
# coding: utf-8

# # CNN using tf.data inpu pipeline bulding API 

import os
from glob import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

import numpy as np
import matplotlib.pyplot as plt
import time


FILE_PATH_LIST = glob('/Users/josea/Desktop/Deep_Learning/Seismic/training/*/*.csv')
DATA = tf.data.Dataset.list_files(FILE_PATH_LIST)


# the load_images function create pairs of seismic record 
# and label (.e., earthquake or noise - 1 or 0)
def load_images(path):
    file = tf.io.read_file(path)
    parts = tf.strings.split(path, os.path.sep)
    bool_values = tf.equal(parts[-2], 'earthquake')
    indices = tf.cast(bool_values, tf.int32)
    return file, indices

ds = DATA.map(load_images).batch(32)

next(iter(ds)) # to check whether the data is properly imported

model = Sequential()

# Input layer + 1st hidden layer
model.add(Conv1D(500, 32))
model.add(Activation('relu'))

# 2nd hidden layer
model.add(Conv1D(250, 32,padding="valid"))
model.add(Activation('relu'))

# 3rd hidden layer
model.add(Conv1D(125, 32))
model.add(Activation('relu'))

# 4th hidden layer
model.add(Conv1D(64, 32))
model.add(Activation('relu'))

# 5th hidden layer
model.add(Conv1D(32, 32))
model.add(Activation('relu'))

# 6th hidden layer
model.add(Conv1D(16, 32))
model.add(Activation('relu'))

# 7th hidden layer
model.add(Conv1D(8, 32))
model.add(Activation('relu'))

# 8th hidden layer
model.add(Conv1D(4, 32))
model.add(Activation('relu'))

# flattening
model.add(Flatten())

# Fully connected layer. 
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

hhistory = model.fit(ds, epochs = 10)

# plot of accuracy as a function of the epochs
plt.plot(hhistory.history['acc'])
plt.plot(hhistory.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('accuracy.png', dpi = 300)
plt.show()

# plot of loss as a function of the epochs
plt.plot(hhistory.history['loss'])
plt.plot(hhistory.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('loss.png', dpi = 300)
plt.show()

