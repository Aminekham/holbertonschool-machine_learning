#!/usr/bin/env python3
"""

"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    
    """
    try:
        pi, m, S = initialize(X, k)
        for i in range(iterations):
            g, l = expectation(X, pi, m, S)
            pi, m, S = maximization(X, g)
            prev_l = l
            _, l = expectation(X, pi, m, S)
            if abs(l - prev_l) <= tol:
                break
            if verbose and i % 10 == 0:
                print(f"Log Likelihood after {i} iterations: {l:.5f}")
        if verbose:
            print(f"Log Likelihood after {i + 1} iterations: {l:.5f}")
        return pi, m, S, g, l
    except Exception as e:
        print(e)
        return None, None, None, None, None
