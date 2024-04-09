#!/usr/bin/env python3
"""

"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    
    """
    try:
        after_t = np.dot(s, np.linalg.matrix_power(P, t))
        return after_t
    except Exception as e:
        return None
