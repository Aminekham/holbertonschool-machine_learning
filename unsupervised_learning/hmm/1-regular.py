#!/usr/bin/env python3
"""
Determening if P represents
a regular markov chain or not
"""
import numpy as np


def regular(P):
    """
    regular markov chain means that we can
    transition from any state to another
    without having any periodic behavior
    """
    try:
        values, right_eigen = np.linalg.eig(P.T)
        idx = np.where(np.isclose(values, 1.0))[0]
        if len(idx) != 1:
            return None
        left_eigen = np.real(right_eigen[:, idx])
        left_eigen /= np.sum(left_eigen)
        return np.array([left_eigen.flatten()])
    except Exception as e:
        return None
