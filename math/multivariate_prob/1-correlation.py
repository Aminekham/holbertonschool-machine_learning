#!/usr/bin/env python3
"""
The correlation matrix
"""
import numpy as np


def correlation(C):
    """
    correlation based on covariance
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    std_dev = np.sqrt(np.diag(C))
    correlation_matrix = np.divide(C, np.outer(std_dev, std_dev), where=std_dev != 0)
    return correlation_matrix