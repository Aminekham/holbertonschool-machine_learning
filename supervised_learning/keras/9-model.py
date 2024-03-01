#!/usr/bin/env python3
"""
saving the model and
loading it from the files
"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    saving the trained model
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    loading the saved model
    """
    model = K.models.load_model(filename)
    return model
