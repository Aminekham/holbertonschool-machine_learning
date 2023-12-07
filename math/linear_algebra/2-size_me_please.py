#!/usr/bin/env python3
def matrix_shape(matrix):
    """
    This module represents a very simple and accurate way to know the shape of a certain matrix with whatever dimensionality
    """
    temp = matrix
    shape = []
    shape.append(len(temp))
    for i in range(len(temp)):
        if type(temp[0]) != list:
            break
        shape.append(len(temp[i]))
        temp = temp[i]
    return(shape)
