#!/usr/bin/env python3
"""
The RMSProp logic
application to fix adagrads optimization
radically diminishing learning rates
"""


import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    using the RMSProp algorithm
    and applying its equations
    """
    p = beta2 * s + (1 - beta2) * grad**2
    new_t = var - (alpha * grad / (np.sqrt(p) + epsilon))
    return new_t, p
