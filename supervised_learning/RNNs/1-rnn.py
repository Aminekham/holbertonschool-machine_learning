#!/usr/bin/env python3
"""
Implementing the RNN
based on the RNNcell from
the previous task
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Full RNN with its all steps
    and iterations
    """
    t, m, _ = X.shape
    h = h_0.shape[1]
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.by.shape[1]))
    H[0] = h_0
    for step in range(t):
        H[step + 1], Y[step] = rnn_cell.forward(H[step], X[step])
    return H, Y
