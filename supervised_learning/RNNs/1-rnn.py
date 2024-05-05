#!/usr/bin/env python3
"""

"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """

    """
    t, m, i = X.shape
    h = h_0.shape[1]
    H = np.zeros((t, m, h))
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))
    h_next = h_0
    for step in range(t):
        h_next, y = rnn_cell.forward(h_next, X[step])
        H[step] = h_next
        Y[step] = y
    return H, Y
