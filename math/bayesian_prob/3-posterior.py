#!/usr/bin/env python3
"""
Computing final posterior
value
"""
import numpy as np
intersection = __import__('1-intersection').intersection
marginal = __import__('2-marginal').marginal


def posterior(x, n, P, Pr):
    """
    Applying the equation of bayesian
    probability
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
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    marginal_value = marginal(x, n, P, Pr)
    intersection_value = intersection(x, n, P, Pr)
    posterior = intersection_value / marginal_value
    return posterior
