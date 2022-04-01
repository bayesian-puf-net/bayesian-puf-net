#!/usr/bin/env python
# coding: utf-8

import os

import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
import scipy as sp
import scipy.stats
import pickle
tfd = tfp.distributions

crps = pickle.load(open('./dapuf_crp.pkl','rb'))

def create_model_conv():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, name='fc1',
                              activation=tf.nn.relu),
        tf.keras.layers.Dense(100, name='fc2',
                              activation=tf.nn.relu),
        tf.keras.layers.Dense(100, name='fc3',
                              activation=tf.nn.relu),
        tf.keras.layers.Dense(100, name='fc4',
                              activation=tf.nn.relu),
        tf.keras.layers.Dense(1, name='fc5',
                              activation=tf.nn.sigmoid)
    ])
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer, loss='binary_crossentropy',
                  metrics=['accuracy'], experimental_run_tf_function=False)
    return model

def create_model_bayes(NUM_TRAIN_EXAMPLES):
    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                              tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))
    model = tf.keras.models.Sequential([
        tfp.layers.DenseFlipout(100, name='fc1',
                                kernel_divergence_fn=kl_divergence_function,
                                activation=tf.nn.relu),
        tf.keras.layers.Dense(100, name='fc2',
                              activation=tf.nn.relu),
        tf.keras.layers.Dense(100, name='fc3',
                              activation=tf.nn.relu),
        tf.keras.layers.Dense(100, name='fc4',
                              activation=tf.nn.relu),
        tf.keras.layers.Dense(1, name='fc5',
                              activation=tf.nn.sigmoid)
    ])
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer, loss='binary_crossentropy',
                  metrics=['accuracy'], experimental_run_tf_function=False)
    return model

np.random.seed(42)

N,C = crps.shape
X = np.zeros((N,C-1), dtype='float32')
for i in range(C-1):
    X[:,i] = crps[:,i:-1].sum(axis=1) % 2
Y = np.zeros(N, dtype='int32')
Y[:] = crps[:,-1]

train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size = 0.2, random_state = 42)
train_labels = train_labels.flatten()
test_labels = test_labels.flatten()

acc,acc_bnn = [],[]
for train_size in [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    print(train_size)
    
    n_train = int(train_features.shape[0] * train_size)

    # conventional
    a = []
    for n in range(20):
        model = create_model_conv()
        model.build(input_shape=(None,32))
        model.fit(train_features[0:n_train], train_labels[0:n_train], epochs=800, batch_size=1000, validation_data=(test_features, test_labels))
        a.append(model.evaluate(test_features, test_labels)[1])
    acc.append(a)
    
    # monte-carlo dropout
    a = []
    for n in range(20):
        model = create_model_bayes(n_train)
        model.build(input_shape=(None,32))
        model.fit(train_features[0:n_train], train_labels[0:n_train], epochs=800, batch_size=1000, validation_data=(test_features, test_labels))
        ll = []
        for m in range(100):
            ll.append(model.predict(test_features).flatten() > 0.5)
        ll = np.array(np.mean(ll,axis=0) > 0.5, dtype='uint8')
        a.append(np.mean(ll == test_labels))
    acc_bnn.append(a)

acc = np.array(acc)
acc_bnn = np.array(acc_bnn)

np.savez('DAPUF_result', acc, acc_bnn)
