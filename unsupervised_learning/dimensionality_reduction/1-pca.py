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
    cov_matrix = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigvals)[::-1]
    sorted_eigvecs = eigvecs[:, sorted_indices]
    top_eigvecs = sorted_eigvecs[:, :ndim]
    Wr = np.dot(X, top_eigvecs)
    return Wr
