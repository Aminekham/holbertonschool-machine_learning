#!/usr/bin/env python3
"""
creating a dropout layer
its widely used for regularization
"""

import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    the dropout work by randomly setting some neurons
    weights to 0 during training to avoid
    overfitingon any used model or
    previous layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    weights = tf.get_variable("weights", shape=[int(prev.shape[1]), n], initializer=init)
    biases = tf.get_variable("biases", shape=[n], initializer=tf.zeros_initializer())
    z = tf.add(tf.matmul(prev, weights), biases)
    if activation is not None:
        a = activation(z)
    else:
        a = z
    dropout = tf.layers.Dropout(rate=1 - keep_prob, training=True)
    output = dropout(a)

    return output
