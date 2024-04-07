#!/usr/bin/env python3
"""

"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """

    """
    try:
        k = pi.shape[0]
        n, d = X.shape
        g = np.zeros((k, n))
        for i in range(k):
            g[i] = pi[i] * pdf(X, m[i], S[i])
        log_likelihood = np.sum(np.log(np.sum(g, axis=0)))
        posterior = g / np.sum(g, axis=0)
        return posterior, log_likelihood
    except Exception as e:
        return None, None
