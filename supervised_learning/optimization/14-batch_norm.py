#!/usr/bin/env python3
"""
batch normalizaiton using tensorflow
"""

import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Create a batch normalization layer with
    given number of features and optional
    activation function
    """
    epsilon = 1e-8
    base = tf.keras.layers.Dense(units=n, kernel_initializer=
                                 tf.keras.initializers.VarianceScaling(
                                     mode='fan_avg'))
    gamma = tf.Variable(tf.ones((1, n)), trainable=True, name='gamma')
    beta = tf.Variable(tf.zeros((1, n)), trainable=True, name='beta')
    Z = base(prev)
    mean, std_div = tf.nn.moments(Z, axes=0)
    Z = tf.nn.batch_normalization(Z, mean, std_div, beta, gamma, epsilon)
    output = activation(Z)
    return output
