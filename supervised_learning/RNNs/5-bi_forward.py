#!/usr/bin/env python3
"""
The bidirectional rnn
"""
import numpy as np


class BidirectionalCell:
    """
    A single cell of the bidirectional rnn
    which can basically learn the data on a two scale
    on forward and backward so it gets the maximum of features
    """
    def __init__(self, i, h, o):
        """
        The init for all the used weights and biases for the forward
        and backward learning
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        the forward which is basically a simple RNN
        hidden forward state
        """
        h_x_concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(h_x_concat, self.Whf) + self.bhf)
        return h_next
