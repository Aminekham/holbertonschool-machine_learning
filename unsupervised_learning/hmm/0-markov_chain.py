#!/usr/bin/env python3
"""
The markov chain
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Calculating the probability to be at a certain state
    after t iterations
    We used the matrix exponentiation because of the nature of
    markov chains and that the current state depends only from
    the previous one which means that raising to the expopent gives
    the needed matrix after the needed t times
    """
    try:
        after_t = np.dot(s, np.linalg.matrix_power(P, t))
        return after_t
    except Exception as e:
        return None
