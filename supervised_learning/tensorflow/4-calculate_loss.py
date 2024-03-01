#!/usr/bin/env python3
"""
Creating the neural network using
tensorflow
"""


import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    calculating the loss with respect to the final predicted
    labels of that step
    """
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                          (labels=y, logits=y_pred))
    return loss
