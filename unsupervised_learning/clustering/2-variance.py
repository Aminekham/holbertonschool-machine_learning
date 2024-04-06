#!/usr/bin/env python3
"""

"""
import numpy as np


def variance(X, C):
    """

    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    if len(X.shape) != 2 or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    if C.shape[0] < 1 or X.shape[0] < 1:
        return None
    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clusters = np.argmin(distances, axis=1)
    variances = np.sum((X - C[clusters])**2)
    return np.sum(variances)
