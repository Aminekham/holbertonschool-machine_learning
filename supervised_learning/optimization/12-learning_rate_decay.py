#!/usr/bin/env python3
"""
learning rate decay over time
using tenserflow compact
"""


import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    tensorflow usage simple and easy
    """
    alpha = tf.train.inverse_time_decay(learning_rate=alpha,
                                        global_step=global_step,
                                        decay_steps=decay_step,
                                        decay_rate=decay_rate,
                                        staircase=True)
    return alpha
