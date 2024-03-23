#!/usr/bin/env python3
"""
The definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    steps and mathematical explanation:
    *testing if the matrix is symetric or not
    *calculating its eigenvalues(How much a vector
    is scaled after applying the matrix transformation)
    *applying the conditions to define a matrix:
    Positive definite: every vector points to the origin
    when transformed by this matrix
    Negative definite: the opposite of Positive definite
    Semi definite: vectors in certain directions are zeros(
    the ellipsoids touch the origin)
    Indefinite: no consistent behavior
    """
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
