#!/usr/bin/env python3
"""
The LSTM implementation
"""
import numpy as np


class LSTMCell:
    """
    single LSTM cell this is the basic building block of an LSTM
    which solves the problem of vanishing gradiants in Traditional RNNs
    """
    def __init__(self, i, h, o):
        """
        The initilization for all the needed parameters for LSTM
        """
        self.Wf = np.random.randn(h + i, h)
        self.bf = np.zeros((1, h))
        self.Wu = np.random.randn(h + i, h)
        self.bu = np.zeros((1, h))
        self.Wc = np.random.randn(h + i, h)
        self.bc = np.zeros((1, h))
        self.Wo = np.random.randn(h + i, h)
        self.bo = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        The forward prob of an LSTM
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        fg = np.dot(concat, self.Wf) + self.bf
        fg = 1 / (1 + np.exp(-fg))
        ug = np.dot(concat, self.Wu) + self.bu
        ug = 1 / (1 + np.exp(-ug))
        c_temp = np.tanh(np.dot(concat, self.Wc) + self.bc)
        c_next = fg * c_prev + ug * c_temp
        og = np.dot(concat, self.Wo) + self.bo
        og = 1 / (1 + np.exp(-og))
        h_next = og * np.tanh(c_next)
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return h_next, c_next, y
