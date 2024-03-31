#!/usr/bin/env python3
"""
The principal component
analysis implementation
"""
import numpy as np


def pca(X, ndim):
    """
    Implementing the pca using the svd generated
    singular and right singular to make it less computationally
    intensive than calculating the covariance matrix and getting the
    eigenvalues decomposition from it
    """
    ls, singular_v, rs = np.linalg.svd(X)
    W = rs.T
    Wr = W[:ndim]
    return Wr
