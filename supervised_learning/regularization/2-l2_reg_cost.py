#!/usr/bin/env python3
"""
calculating the l2 weight regulization
"""

import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """
    regulizing the weights of
    a certain neural network to avoid
    overfitting by adding an l2 regularization term
    """
    l2_losses = tf.losses.get_regularization_losses()
    return cost + l2_losses
