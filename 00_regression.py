#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Train a Fully-Connected Neural Network(MLP) regression on sample data
MLP: Multi-Layer Perceptron
Use mse(mean squared error) loss
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu',input_dim=3))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='mse')
# show Net structure & params
model.summary()

# prepare data
data_x = np.random.normal(0.0, 1.0, (10000, 3))
data_y = 0.2*data_x[:, 0] + 0.3*data_x[:, 1] + 0.95*data_x[:, 2] + 0.25
noise = 0.01*np.random.randn(10000)
data_y = (data_y + noise).reshape(10000, 1)

# split all data to train data & test data
train_data_x = data_x[0:9000]
train_data_y = data_y[0:9000]
test_data_x = data_x[9000:]
test_data_y = data_y[9000:]

# show weights and bias shape
for layer in model.layers:
    weights = layer.get_weights()  # weights(W & b) of each layer are stored in a list
    for weight in weights:
        print(weight.shape)

# train model automatically
# model.fit(train_data_x, train_data_y, batch_size=10, epochs=3)
# train model by steps
for step in range(100):
    cost = model.train_on_batch(train_data_x, train_data_y)
    if step % 10 == 0:
        print("cost:", cost)

# evaluate NN
cost_eval = model.evaluate(test_data_x, test_data_y, batch_size=10)
print("cost_eval:", cost_eval)