#!/usr/bin/env python3
"""

"""
import numpy as np


def pdf(X, m, S):
    """
    
    """
    S = (S + S.T) / 2
    d = len(m)
    det_S = np.linalg.det(S)
    inv_S = np.linalg.inv(S)
    constant = 1 / np.sqrt((2 * np.pi) ** d * det_S)
    exponent = -0.5 * np.sum(np.dot((X - m), inv_S) * (X - m), axis=1)
    P = constant * np.exp(exponent)
    return P
