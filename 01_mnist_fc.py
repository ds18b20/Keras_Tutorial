#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Train a Fully-Connected Neural Network(MLP) classification on MNIST dataset
MLP: Multi-Layer Perceptron
Use categorical_crossentropy loss
"""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

num_classes = 10
batch_size = 23
epochs = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train.shape: (60000, 28, 28)
# y_train.shape: (60000,)
# x_test.shape: (10000, 28, 28)
# y_test.shape: (10000,)
x_train = x_train.reshape(60000, -1)
x_test = x_test.reshape(10000, -1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y=y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y=y_test, num_classes=num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=28*28))  # params count: 28*28*512 + 512(bias)
model.add(Dropout(rate=0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(num_classes, activation='softmax'))

# Show model info
# call .utils.print_summary()
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
