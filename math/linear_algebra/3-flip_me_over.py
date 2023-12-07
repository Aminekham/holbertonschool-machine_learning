#!/usr/bin/env python3
""" This my tranpose function """


def matrix_transpose(matrix):
    """parameters:
            matrix_tranpose: save the final version of each
            column in the matrix
            temp: get the elements in each dimension
                    in a certain position needed
    """
    matrix_transpose = []
    for i in range(len(matrix[0])):
        temp = []
        for j in range(len(matrix)):
            temp.append(matrix[j][i])
        matrix_transpose.append(temp)
    return(matrix_transpose)
