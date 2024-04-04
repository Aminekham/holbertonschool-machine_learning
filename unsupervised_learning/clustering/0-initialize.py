#!/usr/bin/env python3
"""
The k-means init values
to start the algorithm
"""
import numpy as np


def initialize(X, k):
    """
    Using the multivariate uniform
    distribution to have the intial values
    """
    try:
        if k < 0 or not isinstance(k, int):
            return None
        x_min = np.min(X, axis = 0)
        x_max = np.max(X, axis = 0)
        init = np.random.uniform(x_min, x_max, size = (k, X.shape[1]))
        return init
    except:
        return None
