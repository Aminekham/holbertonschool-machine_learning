#!/usr/bin/env python3
"""

"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.by.shape[1]))
    H[0] = h_0
    for step in range(t):
        H[step + 1] = rnn_cell.forward(H[step], X[step])
        Y[step] = np.dot(H[step + 1], rnn_cell.Wy) + rnn_cell.by
    return H[1:], Y
