#!/usr/bin/env python3
"""

"""
import numpy as np


def likelihood(x, n, P):
    def factorial(n):
        if n == 0:
            return 1
        return n * factorial(n - 1)
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if len(P.shape) != 1 or not isinstance(P, np.ndarray):
        raise TypeError("P must be a 1D numpy.ndarray")
    for x in P:
        if x < 0 or x > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
    comb = factorial(n) / (factorial(x)*factorial(n-x))
    likelihood = [comb * p**x * (1-p)**(n-x) for p in P]
    likelihood = np.array(likelihood)
    return likelihood
