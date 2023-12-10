#!/usr/bin/env python3
""" concatinate two matrixs """


def cat_matrices2D(mat1, mat2, axis=0):
    """
    This is a function that concatinates two 2Dmatrixs
    """
    result = []
    if axis == 0 and len(mat1[0]) != len(mat2[0]):
        return None
    elif axis == 1 and len(mat1) != len(mat2):
        return None
    if axis == 0:
        result = [row for row in mat1] + [row for row in mat2]
    elif axis == 1:
        result = []
        for i in range(len(mat1)):
            result.append(mat1[i] + mat2[i])
        return result
    return(result)
