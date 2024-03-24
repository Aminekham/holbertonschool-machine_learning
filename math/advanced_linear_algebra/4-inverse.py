#!/usr/bin/env python3
"""
The inverse of a matrix
"""
adjugate = __import__('3-adjugate').adjugate
determinant = __import__('0-determinant').determinant


def inverse(matrix):
    """
    The inverse which is the
    inverse effect of the transformation
    that the original matrix does
    """
    adjugate_matrix = adjugate(matrix)
    determinant_value = determinant(matrix)
    if determinant_value == 0:
        return None
    inverse_matrix = []
    for i in range(len(adjugate_matrix)):
        row = []
        for j in range(len(adjugate_matrix[0])):
            row.append(adjugate_matrix[i][j] / determinant_value)
        inverse_matrix.append(row)
    return inverse_matrix
