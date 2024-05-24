#!/usr/bin/env python3
import numpy as np
"""

"""


def positional_encoding(max_seq_len, dm):
    """

    """
    positional_encoding_matrix = np.zeros((max_seq_len, dm))
    for pos in range(max_seq_len):
        for i in range(dm):
            angle = pos / np.power(10000, 2 * i / dm)
            if i % 2 == 0:
                positional_encoding_matrix[pos, i] = np.sin(angle)
            else:
                positional_encoding_matrix[pos, i] = np.cos(angle)
    return positional_encoding_matrix
