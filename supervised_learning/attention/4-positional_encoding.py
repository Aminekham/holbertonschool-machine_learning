#!/usr/bin/env python3
import numpy as np
"""

"""


def positional_encoding(max_seq_len, dm):
    """

    """
    pos_encoding = np.zeros((max_seq_len, dm))
    for i in range(dm):
        for pos in range(max_seq_len):
            pos_encoding[pos, i] = pos / np.power(10000, (2 * (i // 2) / dm))
    pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
    return pos_encoding
