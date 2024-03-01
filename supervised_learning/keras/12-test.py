#!/usr/bin/env python3
"""
testing the model
and evaluating it
"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    getting the list of evaluations
    accuracy: how often is the network correct
    loss: how far off are our predictions from the actual values
    """
    acc_loss = network.evaluate(data, labels, verbose=verbose)
    return[acc_loss[0], acc_loss[1]]
