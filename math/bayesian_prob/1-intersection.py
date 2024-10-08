#!/usr/bin/env python3
"""
Computing the intersection
value for the bayesian probability
"""
import numpy as np


def intersection(x, n, P, Pr):
    """
    The intersection is multiplying
    the likelihood by its prior belief
    so we get a weighted likelihood
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    r = "x must be an integer that is greater than or equal to 0"
    if not isinstance(x, int) or x < 0:
        raise ValueError(r)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    for p in P:
        if p < 0 or p > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
    for pr in Pr:
        if pr < 0 or pr > 1:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    if Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    res = 1
    for i in range(x):
        res = res * (n - i) // (i + 1)
    likelihoods = np.zeros_like(P)
    for i, p in enumerate(P):
        likelihoods[i] = res * (p ** x) * ((1 - p) ** (n - x))
    intersection = likelihoods * Pr
    return intersection
