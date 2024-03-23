#!/usr/bin/env python3
"""

"""
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """

    """
    cofactor_matrix = cofactor(matrix)
    adjugate_matrix = []
    for i in range(len(cofactor_matrix[0])):
        tr_row = []
        for j in range(len(cofactor_matrix)):
            tr_row.append(cofactor_matrix[j][i])
        adjugate_matrix.append(tr_row)
    return adjugate_matrix