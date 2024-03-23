#!/usr/bin/env python3
"""

"""
import numpy as np


def definiteness(matrix):
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    transpose = np.transpose(matrix)
    if not np.array_equal(matrix, transpose) or matrix.size == 0:
        return None
    eigenvalues = np.linalg.eigvalsh(matrix)
    if all(eig > 0 for eig in eigenvalues):
        return "Positive definite"
    elif all(eig < 0 for eig in eigenvalues):
        return "Negative definite"
    elif any(eig > 0 for eig in eigenvalues) and any(eigen < 0 
                                                     for eigen in eigenvalues):
        return "Indefinite"
    elif all(eig >= 0 for eig in eigenvalues):
        return "Positive semi-definite"
    elif all(eig <= 0 for eig in eigenvalues):
        return "Negative semi-definite"
