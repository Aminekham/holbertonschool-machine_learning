#!/usr/bin/env python3
"""

"""
import numpy as np


def mean_cov(X):
    if X.shape[1] != 3 or not isinstance(X, np.ndarray):
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0, keepdims=True)
    center = X - mean
    covariance_matrix = np.dot(center.T, center) / (X.shape[0] - 1)
    return mean, covariance_matrix
