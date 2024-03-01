#!/usr/bin/env python3
"""
saving the model
configurations in a json file
"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    saving the configuration
    of the neural network
    as json format
    """
    open(filename, 'w').write({network.get_config()})
    return None

def load_config(filename):
    """
    loading the config of a certain
    model from a json file
    """
    with open(filename) as f:
        config = json.load(f)
    model = K.load_model(config)
    return model
