#!/usr/bin/env python3
"""
training a keras model
in any needed case following certain variables
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """
    Trains the given network with the provided data and labels
    """
    trained = network.fit(x=data, y=labels, batch_size=batch_size,
                          epochs=epochs, verbose=verbose, shuffle=shuffle)
    return trained
