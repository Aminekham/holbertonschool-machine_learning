#!/usr/bin/env python3
"""
Creating the neural network using
tensorflow
"""


import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    performing the training operation
    with reducing the loss value
    with repect to the alpha rate
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    training = optimizer.minimize(loss)
    return training
