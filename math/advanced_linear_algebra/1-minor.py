#!/usr/bin/env python3
"""

"""
determinant = __import__('0-determinant').determinant


def minor(matrix):
    state = all(isinstance(row, list) for row in matrix)
    if matrix == [] or not isinstance(matrix, list) or not state:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]) or matrix[0] == []:
        raise ValueError("matrix must be a non-empty square matrix")
    num_rows = len(matrix)
    minors = []
    if len(matrix) == 1:
        return [[1]]
    for i in range(num_rows):
        minor_row = []
        for j in range(num_rows):
            minor_row.append(determinant([row[:j] + row[j+1:]
                                          for row in matrix[:i] + matrix[i+1:]]))
        minors.append(minor_row)
    return minors
