#!/usr/bin/env python3
"""
using compact tenserflow
to apply within it
the momentum optimizer
"""


import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """
    creating the whole
    momentum optimization process
    """
    opt = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
    return(opt.minimize(loss))
