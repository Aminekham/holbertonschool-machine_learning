#!/usr/bin/env python3
def matrix_shape(matrix):
    temp = matrix
    shape = []
    for i in range(len(temp)):
        shape.append(len(temp[i]))
    return(shape)
