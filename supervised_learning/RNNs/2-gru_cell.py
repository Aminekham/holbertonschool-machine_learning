#!/usr/bin/env python3
"""

"""
import numpy as np


class GRUCell:
    """

    """
    def __init__(self, i, h, o):
        """

        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """

        """
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def softmax(x):
            exps = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exps / np.sum(exps, axis=1, keepdims=True)

        concat = np.concatenate((h_prev, x_t), axis=1)
        z = sigmoid(np.dot(concat, self.Wz) + self.bz)
        r = sigmoid(np.dot(concat, self.Wr) + self.br)
        concat_reset = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.dot(concat_reset, self.Wh) + self.bh)
        h_next = (1 - z) * h_prev + z * h_tilde
        y = softmax(np.dot(h_next, self.Wy) + self.by)
        return h_next, y
