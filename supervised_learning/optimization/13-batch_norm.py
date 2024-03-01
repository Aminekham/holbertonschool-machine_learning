#!/usr/bin/env python3
"""
batch normalization
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    batch normalization using a certain scale
    given by gamma and beta
    """
    mean = np.mean(Z, axis=0)
    std_dev = np.var(Z, axis=0)
    norm = (Z - mean) / (np.sqrt(std_dev + epsilon))
    scaled = norm * gamma + beta
    return scaled
