#!/usr/bin/env python3
""" This my shape getter function """


def matrix_shape(matrix):
    """
    Get the shape of a matrix with any dimensionality.

    Parameters:
    - matrix (list): The input matrix.

    Returns:
    - list: A list representing the shape of
            the matrix. Each element in the list
            corresponds to the size of a dimension.
    """
    temp = matrix
    shape = []
    shape.append(len(temp))
    for i in range(len(temp)):
        if type(temp[0]) != list:
            break
        shape.append(len(temp[i]))
        temp = temp[i]

    return shape
