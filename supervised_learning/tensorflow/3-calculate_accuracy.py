#!/usr/bin/env python3
"""
Creating the neural network using
tensorflow
"""


import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    calculating the mean accuracy
    """
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
