#!/usr/bin/env python3
"""
Calculating the determinant
of a certain matrix
"""

def determinant(matrix):
    """
    The determinant is how much does
    this matrix change in the space
    """
    if not isinstance(matrix, list) or not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]) and matrix[0] != []:
        raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        return det
    num_cols = len(matrix[0])
    det = 0
    for j in range(num_cols):
        minor = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += (-1) ** j * matrix[0][j] * determinant(minor)
    return det
