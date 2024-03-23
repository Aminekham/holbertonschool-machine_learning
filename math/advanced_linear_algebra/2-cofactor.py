#!/usr/bin/env python3
"""

"""
minor = __import__('1-minor').minor


def cofactor(matrix):
    """
    
    """
    minor_matrix = minor(matrix)
    for i in range(len(minor_matrix)):
        for j in range(len(minor_matrix[i])):
            if i % 2 != 0 and j % 2 == 0:
                minor_matrix[i][j] = -1 * minor_matrix[i][j]
            if i % 2 == 0 and j % 2 != 0:
                minor_matrix[i][j] = -1 * minor_matrix[i][j]
    # the final minor matrix is the cofactor matrix
    return minor_matrix
