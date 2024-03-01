#!/usr/bin/env python3
""" mutiply two 2Dmatrix """
shape = __import__('2-size_me_please').matrix_shape


def mat_mul(mat1, mat2):
    """
    multiply two 2D matrixs
    Process:
    1-itterate through the columns of the first matrix
    2- itterate through the rows of the second matrix
    3- itterate through the elemnts of the columns of the first matrix
    and the rows of the second matrix to get the multiplication result
    """
    if shape(mat1)[1] != shape(mat2)[0]:
        return None
    result = []
    for i in range(len(mat1)):
        temp = []
        for j in range(len(mat2[0])):
            element = 0
            for k in range(len(mat2)):
                element += mat1[i][k] * mat2[k][j]
            temp.append(element)
        result.append(temp)
    return(result)
