#!/usr/bin/env python3
"""
applying root mean square propagation
using tenserflow
"""

import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Create RMSprop optimizer"""
    opt = tf.train.RMSPropOptimizer(learning_rate=alpha, epsilon=epsilon,
                                    decay=beta2)
    return opt.minimize(loss)
