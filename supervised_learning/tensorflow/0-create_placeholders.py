#!/usr/bin/env python3
"""
Creating the neural network using
tensorflow
"""


import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    creating the structures to hold the variables
    for the computational graphs
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
