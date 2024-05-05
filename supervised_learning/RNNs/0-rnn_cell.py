#!/usr/bin/env python3
"""
Defining a simple RNN cell
"""
import numpy as np


class RNNCell:
    """
    The RNN cell
    """
    def __init__(self, i, h, o):
        """
        The init function to initalize
        the weights and biases for the hidden states
        and the outputs
        """
        self.Wh = np.random.randn(h + i, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        the forward propagation for the simple cell
        """
        concat_input = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(concat_input, self.Wh) + self.bh)
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y - np.max(y, axis=1, keepdims=True))
        y = y / np.sum(y, axis=1, keepdims=True)
        return h_next, y
