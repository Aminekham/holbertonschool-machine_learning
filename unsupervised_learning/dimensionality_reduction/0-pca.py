#!/usr/bin/env python3
"""
The principal component
analysis implementation
"""
import numpy as np


def pca(X, var=0.95):
    """
    Implementing the pca using the svd generated
    singular and right singular to make it less computationally
    intensive than calculating the covariance matrix and getting the
    eigenvalues decomposition from it
    """
    ls, singular_v, rs = np.linalg.svd(X)
    cum_var = np.cumsum(singular_v) / np.sum(singular_v)
    r = next((i for i, v in enumerate(cum_var) if v >= var)) + 1
    Wr = rs[:r].T
    return Wr
