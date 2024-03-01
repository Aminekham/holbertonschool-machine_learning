#!/usr/bin/env python3
"""
adam optimizer
using tenserflow
"""


import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    create Adam op for training
    """
    opt = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                 beta2=beta2, epsilon=epsilon)
    return opt.minimize(loss)
