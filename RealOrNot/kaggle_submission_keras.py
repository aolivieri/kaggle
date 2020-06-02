# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:28:33 2020

@author: Alexandre
"""


import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras import utils

import pandas as pd

dataset = pd.read_csv('train.csv')

dataset = dataset.drop(['keyword','location'], axis = 1)

train_size = int(len(dataset) * .7)
train_text = dataset['text'][:train_size]
train_target = dataset['target'][:train_size]

test_text = dataset['text'][train_size:]
test_target = dataset['target'][train_size:]

max_words = 1000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_text) # only fit on train

x_train = tokenize.texts_to_matrix(train_text)
x_test = tokenize.texts_to_matrix(test_text)

encoder = LabelEncoder()
encoder.fit(train_target)
y_train = encoder.transform(train_target)
y_test = encoder.transform(test_target)

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

batch_size = 32
epochs = 20

# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])


y_pred = model.predict(x_test)

testset = pd.read_csv('test.csv')

testset = testset.drop(['keyword','location'], axis=1)

max_words = 1000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(testset['text']) 

x_validation = tokenize.texts_to_matrix(testset['text'])

y_validation = model.predict(x_validation)

testset = testset.drop(['text'], axis = 1)

testset['target'] = [1 if x > 0.7 else 0 for x in y_validation[:,0]]

testset.to_csv('kaggle_submission_keras.csv', index = False)

my_submission = pd.read_csv('kaggle_submission_keras.csv')
print(my_submission.head())
print(my_submission.tail())