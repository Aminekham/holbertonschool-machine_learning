#!/usr/bin/env python3
"""
Computing the first term
in the bayesian probability
which is the likelihood
"""
import numpy as np


def intersection(x, n, P, Pr):
    """
    We are having a binomial distribution
    so the as likelihood we will use the PMF:
    P(X=x) = C(n, x)*(P**x)*(1-x)**(n-x)
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if len(P.shape) != 1 or not isinstance(P, np.ndarray):
        raise TypeError("P must be a 1D numpy.ndarray")
    for p in P:
        if p < 0 or p > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    res = 1
    for i in range(x):
        res = res * (n - i) // (i + 1)
    likelihoods = np.zeros_like(P)
    for i, p in enumerate(P):
        likelihoods[i] = res * (p ** x) * ((1 - p) ** (n - x))
    intersection = likelihoods * Pr
    return intersection
