#!/usr/bin/env python3
"""
shuffling the elements
of an np array
"""

import numpy as np


def shuffle_data(X, Y):
    """
    shuffling the elements
    of an np array
    """
    shuffled_indices = np.random.permutation(len(X))
    shuffled_X = X[shuffled_indices]
    shuffled_Y = Y[shuffled_indices]
    return shuffled_X, shuffled_Y
