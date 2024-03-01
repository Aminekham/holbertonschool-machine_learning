#!/usr/bin/env python3
"""
Creating the neural network using
tensorflow
"""


import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    creating a dense layer to give the output
    of our neural network
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(units=n, activation=activation, kernel_initializer=init, name='layer')
    return layer(prev)
