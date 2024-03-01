#!/usr/bin/env python3
"""
calculating the l2 weight regulization
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    regulizing the cost function of
    a certain neural network to avoid
    overfitting by adding an l2 regularization term
    """
    l2_reg = 0
    for i in range(1, L + 1):
        W_key = 'W' + str(i)
        l2_reg += np.sum(np.square(weights[W_key]))
    cost += (lambtha / (2 * m)) * l2_reg
    return cost
