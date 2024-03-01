#!/usr/bin/env python3
"""
predicting the new
input data
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    returning the model predictions
    """
    predictions = network.predict(data, verbose=verbose)
    return predictions
