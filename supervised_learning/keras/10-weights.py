#!/usr/bin/env python3
"""
saving and loading weights and
only the neural network weights
"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    saving the weights
    """
    network.save_weights(filepath=filename)
    return None


def load_weights(network, filename):
    """
    loading the weights
    for the already existing neural
    network
    """
    network.load_weights(filename)
    return None
