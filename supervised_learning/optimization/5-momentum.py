#!/usr/bin/env python3

"""
adam optimization rule
"""


import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    working on gradient descent with momentum
    (applying the moving average to compute some more
    efficient gradients)
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
