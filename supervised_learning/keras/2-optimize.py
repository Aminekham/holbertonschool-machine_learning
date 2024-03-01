#!/usr/bin/env python3
"""
using the adam optimizer
for the already built model
"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    optimizing the network by using
    adam optimizer with the given parameters
    alpha as learning rate, beta1 as the first
    optimization parameter and beta2 as the second
    """
    opt = K.optimizers.Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(optimizer=opt,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return None
