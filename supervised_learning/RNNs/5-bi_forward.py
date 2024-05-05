#!/usr/bin/env python3
"""

"""
import numpy as np


class BidirectionalCell:
    """

    """
    def __init__(self, i, h, o):
        """

        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """

        """
        h_x_concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(h_x_concat, self.Whf) + self.bhf)
        return h_next
