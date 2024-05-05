#!/usr/bin/env python3
"""
The deep RNN
"""
import numpy as np
RNNCell = __import__('0-rnn_cell').RNNCell


def deep_rnn(rnn_cells, X, h_0):
    """
    run a stack of RNN cells on input sequence X
    which gives us the deep rnn
    """
    T, m, i = X.shape
    l, _, h = h_0.shape
    H = np.zeros((T + 1, l, m, h))
    Y = np.zeros((T, m, rnn_cells[-1].Wy.shape[1]))
    H[0] = h_0
    for t in range(T):
        for layer, cell in enumerate(rnn_cells):
            if layer == 0:
                H[t + 1, layer], Y[t] = cell.forward(H[t, layer], X[t])
            else:
                H[t + 1, layer], _ = cell.forward(H[t, layer], H[t + 1, layer - 1])
            Y[t] = Y[t].astype(float)
    return H, Y
