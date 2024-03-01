#!/usr/bin/env python3
""" doc here """
shape = __import__('2-size_me_please').matrix_shape


def add_matrices2D(mat1, mat2):
    """
    A 2Dmatrix adding function
    Parameters:
    temp: temporary list to save the new sub lists each time
    """
    if mat1 == []:
        return([])
    if shape(mat1) != shape(mat2):
        return(None)
    matrix_result = []
    for j in range(len(mat1)):
        temp = []
        for i in range(len(mat1[j])):
            temp.append(mat1[j][i] + mat2[j][i])
        matrix_result.append(temp)
    return(matrix_result)
