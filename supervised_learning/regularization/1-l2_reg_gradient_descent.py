#!/usr/bin/env python3
"""
calculating the l2 weight regulization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    regulizing the weights of
    a certain neural network to avoid
    overfitting by adding an l2 regularization term
    """
    m = Y.shape[1]
    for i in range(1, L + 1):
        W_key = 'W' + str(i)
        b_key = 'b' + str(i)
        A_key = 'A' + str(i)
        A_prev_key = 'A' + str(i - 1)
        diff = cache[A_key] - Y
        if i == L:
            dW = np.dot(diff, cache[A_prev_key].T) / m
        else:
            dW = (np.dot(diff, cache[A_prev_key].T) + lambtha * weights[W_key]) / m
        db = np.sum(diff, axis=1, keepdims=True) / m
        weights[W_key] -= alpha * dW
        weights[b_key] -= alpha * db
    return weights
