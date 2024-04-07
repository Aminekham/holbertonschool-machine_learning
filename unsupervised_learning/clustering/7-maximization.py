#!/usr/bin/env python3
"""

"""
import numpy as np


def maximization(X, g):
    """

    """
    try:
        k, n = g.shape
        d = X.shape[1]
        pi = np.sum(g, axis=1) / n
        m = np.dot(g, X) / np.expand_dims(np.sum(g, axis=1), axis=1)
        S = np.zeros((k, d, d))
        for i in range(k):
            diff = X - m[i]
            S[i] = np.dot(g[i] * diff.T, diff) / np.sum(g[i])
        return pi, m, S
    except Exception as e:
        return None, None, None
